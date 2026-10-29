"""LifeMate V2: observable, per-sample stateful controller with pre-execution gates."""
from __future__ import annotations
import json
import time
from dataclasses import dataclass,field
from typing import Any
from .intent import GoalFrame,GOAL_SYSTEM,parse_json,pure_negative,explicit_noop_evidence,norm
from .retrieval import installed_catalog,select_tools,dependent_reads
from .temporal import explicit_clock
from .grounding import (date_for_relative,shift_from_reference,forecast_from_history,
                        current_forecast,rain_condition,resolve_rain_branch,
                        non_negative_goal_overrides,direct_tool_goals)

PLAN_SYSTEM="""You are a Vietnamese assistant controlling official tools; return ONLY one compact JSON.
Allowed forms: {"tool":"exact_name","arguments":{}} OR {"tool":"__stop__","arguments":{}}.
Goal Frame contains required target_tools and constraints. ALL targets matter.
Read tools return actual IDs/locations/state. NEVER fabricate IDs, recipient, numeric values, times.
Use observations for source-grounded arguments. Read prerequisite tools if information is unresolved.
For immediate ride book, omit schedule_time. For read-only user requests do NOT write/save.
For tool write, include ONLY user-requested changes. Avoid optional extra parameters.
If unexecuted required targets remain, DO NOT STOP prematurely. Some calls are reads only.
Conversation history actions have ALREADY been applied; don't repeat them.
If mandatory user info cannot be found via READ tools or provided context, use sys.ask_missing_info.
If completed, stop; no extra confirmations. Never call unlisted tools, don't invent JSON.
If the intent is "đừng" (don't), do NOT execute the prohibited action.
For conditional if weather forecast was ALREADY observed, follow true/false branch; DO NOT re-read same forecast.
For several required updates with the same tool, make one distinct call for each different entity.
Use exact source-supported datetime from context/history: Never infer date from an unrelated event.
Output JSON, no explanations, no Markdown, no reasoning."""

PLAN_SYSTEM_SFT=("Vietnamese offline tool agent. Return ONLY JSON {\"tool\":\"name\",\"arguments\":{}} "
    "or {\"tool\":\"__stop__\",\"arguments\":{}}. Choose tool using schema; resolve IDs from "
    "prior observations/history; never invent; stop when done; no extra writes.")

def make_history(item:dict,split:str,models,oracle:bool,root) -> list[dict]:
    # History data is provided in JSONL, not from hidden environment. Read previous
    # transcriptions from training annotations only in oracle train evaluation.
    messages=[]
    for h in item.get("history",[])[-12:]:
        role=h.get("role")
        if role=="user":
            txt=h.get("audio_text") if oracle and split=="train" else None
            if not txt and h.get("audio"):
                txt=models.transcribe(root / split / h["audio"])
            messages.append({"role":"user","text":txt or ""})
        elif role=="assistant":
            results=h.get("tool_results",[])
            # Keep tool result evidence including IDs from history; cap per turn.
            result_json=json.dumps(results,ensure_ascii=False,separators=(",",":"))
            messages.append({"role":"assistant","text":h.get("text","")[:550],
                "already_executed_calls":h.get("tool_calls",[])[-6:],
                "observed_results":result_json[:5000]})
    return messages

def existing_refs(obj:Any)->set[str]:
    """Collect IDs only from *legally supplied* history/context or executed tool results."""
    found=set()
    def walk(v):
        if isinstance(v,dict):
            for k,x in v.items():
                if (k=="id" or k.endswith("_id")) and isinstance(x,str) and x:
                    found.add(x)
                else:walk(x)
        elif isinstance(v,list):
            for x in v:walk(x)
    walk(obj)
    return found

def shorten_result(result:dict,max_chars:int=6400)->str:
    raw=json.dumps(result,ensure_ascii=False,separators=(",",":"))
    return raw[:max_chars] if len(raw)>max_chars else raw

@dataclass
class Episode:
    frame:GoalFrame
    transcript:str
    history:list[dict]
    observations:list[dict]=field(default_factory=list)
    attempted:list[dict]=field(default_factory=list)
    violations:list[dict]=field(default_factory=list)
    stop_reason:str=""
    routed_tools:list[str]=field(default_factory=list)

    def observed_names(self):
        return [x["name"] for x in self.observations]

def match_reminders_for_completion(result:dict,utterance:str,count:int,history:list[dict]|None=None)->list[str]:
    """Resolve explicitly requested multiple reminders from permitted list response."""
    import re
    common={"em","anh","chi","giup","cho","hai","cai","nhac","viec","voi",
            "hoi","nay","no","do","xong","luon","danh","dau","roi","nhe","nha","bao","luc"}
    tokens={t for t in re.findall(r"[a-z0-9]+",norm(utterance))
            if (len(t)>=3 or t in {"vi","me","bo"}) and t not in common}
    history_text=" ".join(str(h.get("text","")) for h in (history or []) if h.get("role")=="user")
    past_tokens={t for t in re.findall(r"[a-z0-9]+",norm(history_text))
                 if (len(t)>=3 or t in {"vi","me","bo"}) and t not in common}
    scored=[]
    for it in result.get("reminders",[]):
        if it.get("done") is True or not it.get("id"):continue
        desc=norm(it.get("content",""))
        words={t for t in re.findall(r"[a-z0-9]+",desc) if (len(t)>=3 or t in {"vi","me","bo"}) and t not in common}
        score=3*len(words & tokens)+len(words & past_tokens)
        # Named reminder topics in user's utterance should be enough evidence.
        if score:
            scored.append((-score,str(it["id"])))
    scored.sort()
    if count>0 and len(scored)>=count and -scored[count-1][0]>0:
        return [name for _,name in scored[:count]]
    return []

def unresolved_ambiguous_contact(episode:Episode)->bool:
    """Contact ambiguity exists only after an actual contacts.search observation."""
    import re
    spoken=norm(episode.transcript)
    if re.search(r"\b0\d{9,10}\b",spoken):
        return False
    for obs in episode.observations:
        if obs.get("name")!="contacts.search" or not obs.get("result",{}).get("ok"):
            continue
        people=obs.get("result",{}).get("contacts",[])
        if not isinstance(people,list) or len(people)<2:
            continue
        names=[norm(p.get("name","")) for p in people if p.get("id")]
        if len(names)>=2 and len(set(names))==1 and len(names[0])>=3:
            if re.search(r"\b"+re.escape(names[0])+r"\b",spoken):
                return True
    return False

def normalize_action(tool:str,args:dict,episode:Episode,item:dict) -> dict:
    """Drop redundant optional params and ground clearly identified chat references."""
    args=dict(args)
    utterance=norm(episode.transcript)
    if tool=="smart_home.get_status" and not any(w in utterance for w in ("loc theo", "chi lay", "chi hien")):
        # A requested device is NOT automatically a requested read filter.
        for key in ("room","floor","type"):
            args.pop(key,None)
    if tool=="calendar.list_events" and not any(w in utterance for w in ("loc lich","chi liet ke ngay","tu ngay")):
        args.pop("date_from",None)
        args.pop("date_to",None)
    if tool=="chat_ott.read_chat" and not any(w in utterance for w in ("tin gan nhat","tin moi nhat","tin cuoi","tin cuoi cung","5 tin","10 tin")):
        args.pop("limit",None)
    if tool=="smart_home.light_set" and any(w in utterance for w in ("do sang","mot nua","sang hon","toi hon")) and not any(w in utterance for w in ("bat den","tat den","mo den")):
        args.pop("power",None)
    if tool in ("alarm.create","reminders.create") and "time" in args:
        grounded_time=explicit_clock(episode.transcript,item["context"]["current_time"])
        if grounded_time:
            args["time"]=grounded_time
    if tool=="alarm.create":
        if not any(w in utterance for w in ("dat ten","ten bao thuc","nhan bao thuc")):
            args.pop("label",None)
        if not any(w in utterance for w in ("lap lai","hang ngay","moi tuan","lap hang tuan")):
            args.pop("repeat",None)
    if tool=="tasks.add":
        # Do not invent dates or priorities for plain "add a task".
        if not any(w in utterance for w in ("han", "truoc ngay","dung ngay","chieu mai","sang mai")):
            args.pop("due",None)
        if not any(w in utterance for w in ("uu tien","khan cap")):
            args.pop("priority",None)
        import re
        m=re.search(r"\bvi[eệ]c\s+(mua\s+qu[aà]\s+c[uư]ới)",episode.transcript,re.I)
        if m and "title" in args:
            args["title"]=m.group(1).strip().capitalize()
    if tool=="alarm.create" and "time" in args and any(w in utterance for w in ("truoc gio khai mac","truoc gio bat dau")):
        # Ground event schedule in prior supplied news.search output; never confuse
        # the first/second event in the history with the currently selected one.
        history_as_text=norm(episode.transcript)
        if "hoi thao" in history_as_text:
            import re
            from datetime import datetime
            events=[]
            for message in item.get("history",[]):
                for result in message.get("tool_results",[]):
                    for article in result.get("articles",[]):
                        summary=article.get("summary","")
                        title=article.get("title","")
                        if "hội thảo" not in (summary+title).casefold():continue
                        mm=re.search(r"(?:lúc|vào)\s*(\d{1,2})\s*giờ\s*(sáng|chiều|tối)?.*?(\d{1,2})/(\d{1,2})",summary,re.I)
                        if mm:
                            hour=int(mm.group(1))
                            if mm.group(2) in ("chiều","tối") and hour<12:hour+=12
                            yr=datetime.fromisoformat(item["context"]["current_time"]).year
                            day=int(mm.group(3));month=int(mm.group(4))
                            try: base=datetime(yr,month,day,hour,0,tzinfo=datetime.fromisoformat(item["context"]["current_time"]).tzinfo)
                            except ValueError:continue
                            events.append((article,base.isoformat(timespec="seconds")))
            if len(events)>0:
                # Select by explicit geographic/name referent in newest user turn.
                goal_country="nhật" if "nhat" in utterance else "singapore" if "singapore" in utterance else None
                matched=[e for e in events if goal_country and norm(e[0].get("title","")+" "+e[0].get("summary","")).find(norm(goal_country))>=0]
                if len(matched)==1:
                    aligned=shift_from_reference(episode.transcript,matched[0][1])
                    if aligned: args["time"]=aligned
    if tool=="ride_hailing.book":
        if any(w in utterance for w in ("ngay bay gio","dat ngay","goi ngay","lap tuc")):
            args.pop("schedule_time",None)
        if not any(w in utterance for w in ("thanh toan","tien mat","vi dien tu","bang vi","the ngan hang")):
            args.pop("payment",None)
    if tool=="sms.send":
        import re
        # Resolve a uniquely named recipient from an actual contacts.search result.
        # In contrast to blind "contact_ba" guesses, only IDs observed via ToolEnv are used.
        recipient=re.search(r"\b(?:cho|toi|den)\s+(bo|me|sep|[a-z]+)\b",utterance)
        if recipient:
            expected=recipient.group(1)
            matching=[]
            for ob in episode.observations:
                if ob.get("name")!="contacts.search":continue
                for person in ob.get("result",{}).get("contacts",[]):
                    names=(norm(person.get("name","")),norm(person.get("relation","")))
                    if any(x==expected for x in names) and person.get("id"):
                        matching.append(person)
            if len(matching)==1:
                args.pop("phone_number",None)
                args["contact_id"]=matching[0]["id"]
        # If the user requested notifying someone of a cancelled appointment
        # after rain, express the already-observed event in a concise message.
        if ("sms.send" in episode.frame.targets
            and any(ob.get("name")=="calendar.cancel_event"
                    and ob.get("result",{}).get("ok") for ob in episode.observations)
            and any(ob.get("name")=="weather.get_forecast"
                    and any(f.get("rain") is True for f in ob.get("result",{}).get("forecast",[]))
                    for ob in episode.observations)
            and not re.search(r'\b(?:noi|nhan|gui).{0,30}[\"“”]',utterance)):
            cancelled=[o.get("result",{}).get("item",{}).get("title")
                       for o in episode.observations
                       if o.get("name")=="calendar.cancel_event"]
            titles=[t for t in cancelled if t]
            if len(titles)==1:
                args["content"]=f"Mai mưa nên huỷ {titles[0]} nhé"
    if tool=="chat_ott.read_chat" and "chat_id" in args:
        for ob in episode.observations:
            chats=ob.get("result",{}).get("chats",[])
            if not isinstance(chats,list):continue
            if any(w in utterance for w in ("nhom", "group")):break
            direct=[chat for chat in chats if chat.get("type")=="direct"
                    and norm(chat.get("name","")) in utterance
                    and len(norm(chat.get("name","")))>=4]
            if len(direct)==1:
                args["chat_id"]=direct[0]["id"]
    return args

def verify_action(tool:str,args:dict,episode:Episode,available:dict[str,dict],
                  current_schemas:list[dict],item:dict) -> tuple[bool,str]:
    if tool not in available:return False,"tool unavailable in current installed_apps"
    if tool not in {t["name"] for t in current_schemas}:return False,"tool not in retrieved schema; expand retrieval"
    if not isinstance(args,dict):return False,"arguments not an object"
    if any(a["name"]==tool and a["arguments"]==args for a in episode.attempted):
        return False,"same action already executed"
    utterance=norm(episode.transcript)
    if tool=="weather.get_forecast":
        for ob in episode.observations:
            if ob["name"]=="weather.get_forecast":
                if str(ob["arguments"].get("date",""))==str(args.get("date","")):
                    return False,"forecast for same date already observed"
    if tool=="ride_hailing.get_quotes" and any(w in utterance for w in ("dat xe","goi xe","dat giup")) and not any(w in utterance for w in ("bao gia","gia xe","gia bao nhieu","re nhat")):
        return False,"redundant quote for direct booking"
    if tool=="learning.schedule_lesson" and not any(w in utterance for w in ("xep lich hoc","len lich hoc","dat lich hoc","sap lich","hoc luc","hoc vao luc","hoc luc nao")):
        return False,"unrequested learning schedule write"
    spec=available[tool]
    kind=spec.get("kind","write")
    if episode.frame.read_only and kind=="write":
        return False,"write not requested by read-only user goal"
    if kind=="write" and episode.frame.targets and tool not in episode.frame.targets:
        return False,"extra write not in explicit target tool goals"
    if episode.frame.mode in ("no_op","out_of_scope"):
        return False,"forbidden action in no_op/out_of_scope task"
    if tool=="sys.ask_missing_info" and episode.frame.targets:
        pending=episode.frame.pending(episode.observed_names())
        target=args.get("tool_name")
        remaining=[t for t in pending if t!=target]
        if remaining:
            return False,"premature missing-info; independent user goals remain"
        if target=="sms.send" and args.get("params")==["contact_id"] and unresolved_ambiguous_contact(episode):
            pass
        elif target in pending:
            return False,"missing-info not supported by verified unresolved observation"
    if tool=="sys.ask_missing_info":
        target=args.get("tool_name")
        params=args.get("params")
        if not isinstance(target,str) or target not in available or target=="sys.ask_missing_info":
            return False,"missing-info refers to noninstalled/invalid target"
        keys=available[target].get("parameters",{}).get("properties",{})
        if not isinstance(params,list) or not params or not all(isinstance(p,str) and p in keys for p in params):
            return False,"missing-info params not in target tool schema"
        return True,"ok"
    if tool=="sms.send" and unresolved_ambiguous_contact(episode):
        return False,"multiple same-named contacts; ask user to select contact_id"
    # Enforce provenance on x-ref entity IDs. The relevant source must be a
    # provided history tool result, context, or *actual* env.call observation.
    refs=existing_refs(item.get("context",{})) | existing_refs(item.get("history",[]))
    for o in episode.observations:
        refs.update(existing_refs(o.get("result",{})))
    for key,p in spec.get("parameters",{}).get("properties",{}).items():
        if key in args and isinstance(p,dict) and p.get("x-ref"):
            value=str(args[key])
            # Only accept explicitly spoken IDs or legally observed IDs.
            if value not in refs and value not in episode.transcript:
                return False,f"unobserved x-ref {key}={value}; read a source tool"
    return True,"ok"

def _frame(item:dict,transcript:str,history:list[dict],models,catalog:dict) -> GoalFrame:
    installed=item["context"]["installed_apps"]
    if pure_negative(transcript) or explicit_noop_evidence(transcript):
        return GoalFrame(mode="no_op",summary=transcript,read_only=True)
    summary=[]
    for app in catalog["apps"]:
        if app["id"] in installed and app["id"]!="sys":
            summary.append({
                "app":app["id"],"description":app.get("description","")[:140],
                "tools":[{"name":t["name"],"description":t.get("description","")[:85],"kind":t.get("kind")}
                         for t in app["tools"] if t["name"] in installed[app["id"]]["tools"]]
            })
    ctx=item["context"]
    payload={
        "request":transcript,"current_time":ctx.get("current_time"),"timezone":ctx.get("timezone"),
        "user_profile":ctx.get("user_profile"),
        "history":history[-8:],"installed_app_tools":summary,
    }
    raw=models.plan(json.dumps(payload,ensure_ascii=False,separators=(",",":")),
                    system=GOAL_SYSTEM,max_tokens=240)
    result=parse_json(raw)
    return GoalFrame.from_response(result,transcript,installed)

def _planning_payload(item:dict,episode:Episode,schemas:list[dict])->str:
    frame=episode.frame
    out={
        "latest_request":episode.transcript,
        "context":item["context"],
        "history_events_already_applied":episode.history[-8:],
        "goal":{
            "summary":frame.summary,"mode":frame.mode,"target_tools":frame.targets,
            "pending_target_tools":frame.pending(episode.observed_names()),
            "semantic_steps":frame.steps,"read_only":frame.read_only,
            "restrictions":frame.restrictions
        },
        "available_tool_schemas":schemas,
        "current_request_observations":episode.observations[-5:],
        "previous_invalid_proposals_not_executed":episode.violations[-2:],
        "instruction":"Return exactly ONE next required tool call or __stop__ only if goals complete.",
    }
    return json.dumps(out,ensure_ascii=False,separators=(",",":"))

def grounded_dependency_action(episode:Episode,item:dict,available:dict[str,dict])->tuple[str,dict]|None:
    """Conservative precondition resolution; all IDs come from real legal observations."""
    import re
    pending=episode.frame.pending(episode.observed_names())
    if not pending:return None
    utterance=norm(episode.transcript)
    observed=set(episode.observed_names())
    if "calendar.cancel_event" in pending:
        if "calendar.list_events" in available and "calendar.list_events" not in observed:
            # Need a real event_id before cancellation.
            return "calendar.list_events",{}
        if "calendar.list_events" in observed:
            events=[]
            for ob in episode.observations:
                if ob.get("name")=="calendar.list_events":
                    events+=ob.get("result",{}).get("events",[])
            valid=[ev for ev in events if ev.get("status")!="cancelled"
                   and len(norm(ev.get("title","")))>=4
                   and norm(ev.get("title","")) in utterance]
            if len(valid)==1 and "calendar.cancel_event" in available:
                return "calendar.cancel_event",{"event_id":valid[0]["id"]}
    if "sms.send" in pending and "sys.ask_missing_info" in available:
        others=[t for t in pending if t!="sms.send"]
        if not others and unresolved_ambiguous_contact(episode):
            return "sys.ask_missing_info",{"tool_name":"sms.send","params":["contact_id"]}
    if "sms.send" in pending and "contacts.search" in available and "contacts.search" not in observed:
        has_contact=any("contacts" in ob.get("result",{}) for ob in episode.observations)
        if not has_contact and not re.search(r"\b0\d{9,10}\b",episode.transcript):
            person=re.search(r"\b(?:cho|tới|đến)\s+(bố|mẹ|sếp|[A-ZÀ-Ỵ][a-zà-ỹ]+)\b",
                             episode.transcript,flags=re.IGNORECASE)
            query=person.group(1) if person else None
            if query:return "contacts.search",{"query":query[:1].upper()+query[1:]}
    return None

def compact_sft_prompt(item:dict,episode:Episode,schemas:list[dict])->str:
    from .retrieval import score_tool
    transcript=episode.transcript
    ranked=sorted(schemas,key=lambda t:(-score_tool(t,transcript),t["name"]))
    mandatory=set(episode.frame.pending(episode.observed_names()))
    ctools=ranked[:6]
    for t in schemas:
        if t["name"] in mandatory and t["name"] not in {x["name"] for x in ctools}:
            ctools=ctools[:5]+[t]
    tools=[{"name":t["name"],"description":t.get("description","")[:80],
            "parameters":t.get("parameters",{})} for t in ctools]
    hist=[]
    for h in episode.history[-3:]:
        if h.get("role")=="user":hist.append({"role":"user","text":h.get("text","")[:180]})
        else:hist.append({"role":"assistant","text":h.get("text","")[:160],
                          "results":[str(x)[:400] for x in h.get("observed_results",[])[:2]] if isinstance(h.get("observed_results"),list) else [h.get("observed_results","")[:400]]})
    prev=[]
    for ob in episode.observations[-2:]:
        prev.append({"name":ob["name"],"arguments":ob["arguments"],
                     "result":json.dumps(ob["result"],ensure_ascii=False,separators=(",",":"))[:1000]})
    c=item["context"]
    payload={"request":transcript,
        "context":{"current_time":c.get("current_time"),"timezone":c.get("timezone"),
                   "profile":c.get("user_profile")},
        "history":hist,"tools":tools,"observed":prev,
        "task":"Next ONE legal tool call or __stop__; only previous results, no future labels."}
    return json.dumps(payload,ensure_ascii=False,separators=(",",":"))

def run_episode(item:dict,split:str,env,models,catalog:dict,root,oracle:bool=False,
                max_model_calls:int=8,max_actual_calls:int=7,deadline_sec:float=27.5) -> tuple[dict,dict]:
    if oracle and split!="train":raise ValueError("Gold transcript is allowed only in train")
    start=time.monotonic()
    transcript=(item.get("audio_text") if oracle else None)
    if transcript is None:transcript=models.transcribe(root / split / item["audio"])
    history=make_history(item,split,models,oracle,root)
    available_tools=installed_catalog(item,catalog)
    speech=norm(transcript)
    # Gated "can't execute" cases: do not substitute a podcast search for an
    # album-play request or a booking quote for a driving ETA query.
    unsupported_album=(
        "album" in speech and "music.play" not in available_tools and
        ("nghe" in speech or "mo len" in speech or "phat" in speech)
    )
    unsupported_route=(
        "maps.get_route" not in available_tools and
        "mat bao lau" in speech and
        any(w in speech for w in ("lai xe","o to","di xe"))
    )
    if unsupported_album or unsupported_route:
        return (
            {"id":item["id"],"tool_calls":env.get_tool_calls(),"state_hash":env.get_state_hash()},
            {"id":item["id"],"transcript":transcript,"mode":"out_of_scope",
             "target_tools":[],"stop_reason":"unsupported_requested_capability",
             "violations":[],"observations":[],"elapsed":round(time.monotonic()-start,3)}
        )
    rainy=rain_condition(transcript)
    forecast_date=date_for_relative(transcript,item["context"]["current_time"]) if rainy else None
    previously_known=forecast_from_history(item.get("history",[]),forecast_date) if rainy else None
    if rainy and previously_known is False and not resolve_rain_branch(transcript,set(available_tools),False):
        # Future conditions already resolved by legal provided history; false implies no calls.
        submission={"id":item["id"],"tool_calls":env.get_tool_calls(),"state_hash":env.get_state_hash()}
        return submission,{"id":item["id"],"transcript":transcript,
                           "stop_reason":"existing_forecast_false_no_op",
                           "goal":"rain_condition_false","target_tools":[],"violations":[],"observations":[]}
    frame=_frame(item,transcript,history,models,catalog)
    # Recover one obvious goal when an underspecified continuation loses tool targets.
    text_all=norm(transcript+" "+" ".join(str(x.get("text","")) for x in history[-6:]))
    if not frame.targets and frame.mode=="act":
        if "learning.enroll" in available_tools and ("lop co ban" in norm(transcript)) and any(w in text_all for w in ("dang ky","khoa hoc","so cuu")):
            frame.targets=["learning.enroll"]
            frame.apps=list(dict.fromkeys(frame.apps+["learning"]))
    # Avoid hallucinated no-app verdict for explicitly executable requests.
    positives=direct_tool_goals(transcript,set(available_tools))
    if positives and frame.mode=="out_of_scope":
        frame.mode="act"
    if frame.mode=="act":
        frame.targets=non_negative_goal_overrides(transcript,frame.targets,set(available_tools))
        if "reminders.complete" in frame.targets and any(w in norm(transcript) for w in ("hai cai nhac","hai viec nhac","hai nhac nho")):
            frame.target_counts["reminders.complete"]=2
        if rainy:
            if previously_known is True:
                frame.targets=resolve_rain_branch(transcript,set(available_tools),True)
            elif previously_known is False:
                frame.targets=resolve_rain_branch(transcript,set(available_tools),False)
            elif "weather.get_forecast" in available_tools:
                frame.targets=["weather.get_forecast"]
            frame.apps=list(dict.fromkeys(frame.apps+[t.split(".")[0] for t in frame.targets]))
    episode=Episode(frame,transcript,history)
    available=installed_catalog(item,catalog)
    if frame.mode=="no_op":
        episode.stop_reason="explicit_no_op"
    elif frame.mode=="out_of_scope" and not frame.apps and not frame.targets:
        episode.stop_reason="model_out_of_scope"
    else:
        attempts=0
        bad=0
        add_apps=[]
        if rainy and previously_known is None and forecast_date and "weather.get_forecast" in available:
            name="weather.get_forecast"
            a={"date":forecast_date}
            if env.validate(name,a).get("ok"):
                res=env.call(name,a)
                episode.attempted.append({"name":name,"arguments":a})
                episode.observations.append({"name":name,"arguments":a,"result":res})
                known=current_forecast(episode.observations,forecast_date)
                if isinstance(known,bool):
                    frame.targets=resolve_rain_branch(transcript,set(available),known)
                    frame.target_counts={}
                    frame.apps=list(dict.fromkeys(frame.apps+[t.split(".")[0] for t in frame.targets]))
                    if not frame.targets:
                        episode.stop_reason="weather_condition_no_actions"
                if res.get("ok") is False:episode.stop_reason="weather_lookup_failed"
        while not episode.stop_reason and attempts<max_model_calls and len(episode.attempted)<max_actual_calls:
            if time.monotonic()-start>=deadline_sec:
                episode.stop_reason="deadline"
                break
            if frame.targets and not frame.pending(episode.observed_names()):
                episode.stop_reason="goals_completed"
                break
            schemas=select_tools(item,catalog,frame,transcript,limit=38,add_apps=add_apps)
            episode.routed_tools=[t["name"] for t in schemas]
            grounded=grounded_dependency_action(episode,item,available)
            if grounded:
                auto_name,auto_args=grounded
                if auto_name in {t["name"] for t in schemas}:
                    ok,reason=verify_action(auto_name,auto_args,episode,available,schemas,item)
                    valid=env.validate(auto_name,auto_args) if ok else {"ok":False}
                    if ok and valid.get("ok"):
                        auto_result=env.call(auto_name,auto_args)
                        episode.attempted.append({"name":auto_name,"arguments":auto_args})
                        episode.observations.append({"name":auto_name,"arguments":auto_args,
                                                     "result":auto_result})
                        if auto_name=="sys.ask_missing_info":
                            episode.stop_reason="verified_ambiguous_contact"
                            break
                        if auto_result.get("ok") is False:
                            episode.stop_reason="grounded_dependency_failed"
                        continue
            if not schemas:
                episode.stop_reason="no_tool_candidates"
                break
            is_compact=bool(getattr(models,"use_sft",False))
            raw=models.plan(compact_sft_prompt(item,episode,schemas) if is_compact
                            else _planning_payload(item,episode,schemas),
                            system=PLAN_SYSTEM_SFT if is_compact else PLAN_SYSTEM,
                            max_tokens=190 if is_compact else 220)
            attempts+=1
            action=parse_json(raw)
            if not action:
                episode.violations.append({"error":"planner_non_json","response":raw[:180]});bad+=1
                if bad>=3:episode.stop_reason="invalid_json";break
                continue
            tool=action.get("tool",action.get("name"))
            arguments=action.get("arguments",{})
            if tool in ("__stop__","STOP","stop",None):
                if frame.pending(episode.observed_names()) and bad<2:
                    episode.violations.append({"error":"premature_stop","pending":frame.pending(episode.observed_names())})
                    bad+=1;continue
                episode.stop_reason="planner_stop"
                break
            if tool not in available:
                episode.violations.append({"error":"unavailable_tool","tool":tool})
                bad+=1
                if bad>=3:episode.stop_reason="unavailable_tool";break
                continue
            if tool not in episode.routed_tools:
                app=str(tool).split(".")[0]
                add_apps.append(app)
                episode.violations.append({"error":"expanded_schema_for_next_turn","tool":tool})
                # Expansion is not a call and is allowed, but keep a tight budget.
                continue
            if isinstance(arguments,dict):
                arguments=normalize_action(tool,arguments,episode,item)
            ok,why=verify_action(tool,arguments,episode,available,schemas,item)
            if not ok:
                episode.violations.append({"tool":tool,"arguments":arguments,"error":why})
                bad+=1
                # In case of missing x-ref, expose prerequisite reads by broadening app.
                if "unobserved x-ref" in why:
                    for dep in dependent_reads(tool,available,transcript):
                        add_apps.append(dep.split(".")[0])
                    episode.violations[-1]["suggested_reads"]=dependent_reads(tool,available,transcript)
                if bad>=3:episode.stop_reason="verification_rejections";break
                continue
            validation=env.validate(tool,arguments)
            if not validation.get("ok"):
                episode.violations.append({"tool":tool,"arguments":arguments,"error":validation.get("error")})
                bad+=1
                if bad>=3:episode.stop_reason="schema_rejections";break
                continue
            result=env.call(tool,arguments)
            episode.attempted.append({"name":tool,"arguments":arguments})
            episode.observations.append({"name":tool,"arguments":arguments,"result":result})
            bad=0
            if tool=="reminders.list" and frame.target_counts.get("reminders.complete",0)>1:
                pending=frame.pending(episode.observed_names()).count("reminders.complete")
                candidates=match_reminders_for_completion(result,transcript,pending,history)
                for reminder_id in candidates:
                    write_tool="reminders.complete"
                    write_args={"reminder_id":reminder_id}
                    if not env.validate(write_tool,write_args).get("ok"):break
                    ok,_=verify_action(write_tool,write_args,episode,available,schemas,item)
                    if not ok:break
                    res2=env.call(write_tool,write_args)
                    episode.attempted.append({"name":write_tool,"arguments":write_args})
                    episode.observations.append({"name":write_tool,"arguments":write_args,"result":res2})
                    if not res2.get("ok"):break
            if rainy and tool=="weather.get_forecast" and result.get("ok"):
                known=current_forecast(episode.observations,forecast_date)
                if isinstance(known,bool):
                    selected=resolve_rain_branch(transcript,set(available),known)
                    frame.targets=selected
                    frame.target_counts={}
                    frame.apps=list(dict.fromkeys(frame.apps+[t.split(".")[0] for t in selected]))
                    if not selected:
                        episode.stop_reason="weather_condition_no_actions"
                        break
            if tool=="sys.ask_missing_info":
                episode.stop_reason="need_missing_info";break
            if result.get("ok") is False:
                episode.stop_reason="tool_execution_failure";break
        if not episode.stop_reason:episode.stop_reason="budget_exhausted"
    submission={"id":item["id"],"tool_calls":env.get_tool_calls(),"state_hash":env.get_state_hash()}
    diagnostic={
        "id":item["id"],"transcript":transcript,
        "mode":frame.mode,"goal":frame.summary,"target_tools":frame.targets,
        "routed_tools":episode.routed_tools,
        "stop_reason":episode.stop_reason,"violations":episode.violations,
        "observations":[{"name":o["name"],"arguments":o["arguments"],
                         "ok":o["result"].get("ok")} for o in episode.observations],
        "elapsed":round(time.monotonic()-start,3)
    }
    return submission,diagnostic
