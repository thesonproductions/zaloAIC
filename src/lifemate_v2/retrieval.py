"""LifeMate V2: dynamic, catalog-grounded tool selection and prerequisites."""
from __future__ import annotations
import json
import re
from .intent import GoalFrame,norm

# These are generic Vietnamese capability aliases, NOT test labels or hardcoded IDs.
APP_TERMS={
    "smart_home":("den","sang","quat","dieu hoa","phong","thiet bi","scenes"),
    "calendar":("lich","cuoc hop","hop phu huynh","su kien","hen","thoi gian"),
    "ride_hailing":("xe","dat xe","don","dua","taxi","cuoc xe"),
    "contacts":("lien he","ten nguoi","danh ba","so dien thoai","sep","me"),
    "sms":("sms","nhan tin","gui tin","tin nhan"),
    "chat_ott":("zalo","chat","tin nhan zalo","nhan zalo","nhom"),
    "news":("tin tuc","tin moi","khoa hoc","thoi su","doc tin"),
    "learning":("lop","khoa hoc","dang ky hoc","mon hoc","cong dong"),
    "weather":("thoi tiet","mua","du bao","nhiet do"),
    "alarm":("bao thuc","dat gio","nhac gio"),
    "food_delivery":("do an","mon an","bun","pho","goi mon","dat do an"),
    "medication":("thuoc","uong thuoc","khang sinh"),
    "phone":("goi dien","cuoc goi"),
    "reminders":("nhac nho","nhac viec"),
    "video_meeting":("phong hop","hop truc tuyen","meeting","join"),
}
def compact(tool:dict) -> dict:
    return {"name":tool["name"],"kind":tool.get("kind","write"),
            "description":tool.get("description","")[:260],
            "parameters":tool.get("parameters",{}),"returns":tool.get("returns","")}

def installed_catalog(item:dict,catalog:dict)->dict[str,dict]:
    allowed={t for x in item["context"]["installed_apps"].values() for t in x["tools"]} | {"sys.ask_missing_info"}
    return {t["name"]:t for app in catalog["apps"] for t in app["tools"] if t["name"] in allowed}

def score_tool(tool:dict,query:str,app_profile:dict|None=None)->float:
    """Schema-grounded lexical retriever; no train/test label or hidden app state."""
    nq=norm(query)
    desc=norm(tool["name"]+" "+tool.get("description",""))
    words={x for x in re.findall(r"[a-z0-9]+",nq) if len(x)>2}
    relevance=len(words & set(re.findall(r"[a-z0-9]+",desc)))
    app=tool["name"].split(".")[0]
    for term in APP_TERMS.get(app,()):
        t=norm(term)
        if re.search(r"\b"+re.escape(t)+r"\b",nq):
            relevance+=4
    if app in nq or app.replace("_"," ") in nq:relevance+=5
    if app_profile:
        metadata=norm(" ".join(str(app_profile.get(k,"")) for k in ("name_vi","description","group")))
        meta_words={x for x in re.findall(r"[a-z0-9]+",metadata) if len(x)>2}
        relevance+=2*len(words & meta_words)
    return relevance

def dependent_reads(target:str,available:dict[str,dict],transcript:str)->list[str]:
    tool=available.get(target)
    if not tool:return []
    app=target.split(".")[0]
    deps=[]
    for name,other in available.items():
        if other.get("kind")!="read":continue
        if name==target:continue
        # Catalog references and explicit ID provenance.
        for prop in tool.get("parameters",{}).get("properties",{}).values():
            ref=prop.get("x-ref","")
            desc=norm(str(prop.get("description","")))
            ret=other.get("returns","")
            if ref and (name.split(".")[0]==ref.split(".")[0]) and (
                ref.split(".")[-1] in ret or name.split(".")[0] in desc
            ):
                deps.append(name)
    # Domain-independent-ish task-time dependency hints; legal tool availability checked.
    text=norm(transcript)
    if target=="ride_hailing.book" and any(w in text for w in ("hop","su kien","hen")) and "calendar.list_events" in available:
        deps.append("calendar.list_events")
    if target=="chat_ott.read_chat" and "chat_ott.list_chats" in available:
        deps.append("chat_ott.list_chats")
    if target=="learning.enroll" and "learning.list_courses" in available:
        deps.append("learning.list_courses")
    if target=="smart_home.light_set" and "smart_home.get_status" in available:
        deps.append("smart_home.get_status")
    if target=="sms.send" and "contacts.search" in available:
        deps.append("contacts.search")
    return list(dict.fromkeys(deps))

def select_tools(item:dict,catalog:dict,frame:GoalFrame,transcript:str,limit:int=30,
                 add_apps:list[str]|None=None) ->list[dict]:
    available=installed_catalog(item,catalog)
    if not available:return []
    app_meta={app["id"]:app for app in catalog["apps"]}
    apps=set(frame.apps) | set(add_apps or [])
    for t in frame.targets:apps.add(t.split(".")[0])
    if not apps:
        scores={}
        for name,t in available.items():
            app=name.split(".")[0]
            scores[app]=max(scores.get(app,0),score_tool(t,transcript+" "+frame.summary,app_meta.get(app)))
        apps={k for k,v in sorted(scores.items(),key=lambda x:(-x[1],x[0]))[:5] if v>0}
        # When a message asks to contact a named person, provide the catalog's
        # read tools for resolving a contact, without calling them yet.
        if "sms" in apps and "contacts.search" in available:apps.add("contacts")
        if "chat_ott" in apps and "chat_ott.list_chats" in available:apps.add("chat_ott")

    targets={x for x in frame.targets if x in available}
    deps=set()
    for target in targets:deps.update(dependent_reads(target,available,transcript))
    apps |= {t.split(".")[0] for t in deps}
    candidates=[t for t in available.values() if t["name"].split(".")[0] in apps]
    rank=lambda t:(
        -int(t["name"] in targets), -int(t["name"] in deps),
        -score_tool(t,transcript+" "+frame.summary,app_meta.get(t["name"].split(".")[0])),
        t["name"]
    )
    candidates=sorted(candidates,key=rank)[:limit]
    # Keep room for cross-cutting missing-info action.
    if "sys.ask_missing_info" in available and not any(t["name"]=="sys.ask_missing_info" for t in candidates):
        candidates.append(available["sys.ask_missing_info"])
    return [compact(t) for t in candidates]

def all_schema_names(item:dict,catalog:dict)->set[str]:
    return set(installed_catalog(item,catalog))
