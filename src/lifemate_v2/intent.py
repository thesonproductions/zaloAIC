"""LifeMate V2: goal/polarity parser and dialogue context (no direct state access)."""
from __future__ import annotations
import json
import re
import unicodedata
from dataclasses import dataclass, field

def norm(text: str) -> str:
    text=unicodedata.normalize("NFD",str(text).casefold()).replace("đ","d")
    return "".join(c for c in text if not unicodedata.combining(c))

def parse_json(text: str) -> dict | None:
    s=text.strip()
    if s.startswith("```"):
        s=re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$","",s)
    try:
        x=json.loads(s)
        return x if isinstance(x,dict) else None
    except ValueError:
        pass
    decoder=json.JSONDecoder()
    for i,char in enumerate(s):
        if char=="{":
            try:
                x,_=decoder.raw_decode(s[i:])
                if isinstance(x,dict):return x
            except ValueError:
                continue
    return None

def pure_negative(text: str) -> bool:
    """True only for simple 'DO NOT change' requests. A 'STOP/turn off' request is positive."""
    s=text.casefold().strip(" ,.!?\t\n")
    s=re.sub(r"^(em ơi|trợ lý ơi|anh ơi|chị ơi|này em|ơi|dạ|em)\s*[,!]?[ ]*","",s)
    negative=bool(re.match(r"^(đừng|không cần|khỏi|chớ)\b",s))
    if not negative:return False
    # "Don't forget to..." commands DO request a real action.
    if re.match(r"^đừng\s+quên\b",s):return False
    mixed=re.search(r"\b(nhưng|thay vào đó|rồi|sau đó|đồng thời|hãy|mà|và bật|và mở|và chuyển|và gửi)\b",s)
    conditional=re.search(r"\b(nếu|thì|còn nếu)\b",s)
    return not bool(mixed or conditional) and len(s)<165

def reading_only(text: str) -> bool:
    """Detect information-only requests; mixed read + write is not read-only."""
    s=norm(text)
    read=bool(re.search(r"\b(xem|doc|tim|tra cuu|kiem tra|cho biet|co tin|tin gi|mo tin nhan|danh sach|bao nhieu|gia|so du)\b",s))
    write=bool(re.search(r"\b(luu|dat|mua|thanh toan|gui|nhan cho|nhan tin cho|giao|dong|nhac|tao|cap nhat|doi|chuyen|bat|tat|sua|huy|xoa|dang ky|tham gia|goi xe|them|dat cho|danh dau)\b",s))
    if re.search(r"\b(khóa cửa|khóa lại|khóa nó|dừng báo thức)\b",text.casefold()):
        write=True
    return read and not write

def explicit_noop_evidence(text:str)->bool:
    """Conservative discourse-level no-op/retraction recognition.
    Do NOT treat 'khỏi giữ ấm' as cancelling a separate 'tắt nồi cơm'.
    """
    s=norm(text)
    # Terminal withdrawal of a request.
    if re.search(r"(a|ah|thoi)\s+thoi\s+khoi\b|\b(a|thoi)\s+khoi\b",s):
        if not re.search(r"\b(roi|sau do|nhung|con)\s+\w+\s+(bat|tat|dat|gui)\b",s):
            return True
    # Pure acknowledgement with no new action request.
    if re.search(r"\b(cam on|de.*ban.*da|chuc.*vui ve)\b",s):
        if not re.search(r"\b(dat|gui|tao|nhac|xem|bat|tat|khoa|huy|xoa|tra cuu|tham gia|thanh toan|doi phieu|nhan cho|nhan tin)\b",s):
            return True
    return False


@dataclass
class GoalFrame:
    mode: str="act"
    summary: str=""
    apps: list[str]=field(default_factory=list)
    targets: list[str]=field(default_factory=list)
    target_counts:dict[str,int]=field(default_factory=dict)
    steps: list[str]=field(default_factory=list)
    read_only: bool=False
    restrictions: list[str]=field(default_factory=list)
    raw: dict=field(default_factory=dict)

    def pending(self, called: list[str]) -> list[str]:
        from collections import Counter
        done=Counter(called)
        remaining=[]
        for name in self.targets:
            needed=min(5,max(1,int(self.target_counts.get(name,1))))
            remaining.extend([name]*max(0,needed-done[name]))
        return remaining

    @classmethod
    def from_response(cls, response: dict | None, transcript: str, installed: dict) -> "GoalFrame":
        x=response or {}
        mode=x.get("mode","act")
        if mode not in ("act","no_op","out_of_scope"):mode="act"
        # Deterministic safety: simple imperative negations take priority.
        if pure_negative(transcript):mode="no_op"
        apps=[a for a in x.get("apps",[]) if isinstance(a,str) and a in installed] if isinstance(x.get("apps"),list) else []
        targets=x.get("target_tools",[])
        if not isinstance(targets,list):targets=[]
        allowed={t for app in installed.values() for t in app.get("tools",[])}
        targets=list(dict.fromkeys(t for t in targets if isinstance(t,str) and t in allowed))
        # User explicitly forbids a specific sub-action. Drop only that
        # target, never unrelated positive actions in the same request.
        utterance_norm=norm(transcript)
        if re.search(r"\bquat\b.{0,26}\b(khong can|dung|khoi)\s*(bat|chay)?\b",utterance_norm):
            targets=[t for t in targets if t!="smart_home.fan_set"]
        # No confident OOS claim merely because router selected nothing.
        if mode=="out_of_scope" and apps:mode="act"
        # Reject unsupported model assertions of no-op. Only explicit
        # negation, user retraction or pure social acknowledgement justify it.
        if mode=="no_op" and not (pure_negative(transcript) or explicit_noop_evidence(transcript)):
            mode="act"
        counts={}
        proposed=x.get("target_counts",{})
        if isinstance(proposed,dict):
            counts={k:min(5,max(1,int(v))) for k,v in proposed.items()
                    if k in targets and isinstance(v,int)}
        # Quantity for repetitive effects. E.g. "hai cái nhắc ... đánh dấu xong".
        s=norm(transcript)
        if len(targets)==1 and any(w in s for w in ("hai cai nhac","hai viec nhac","hai nhac nho","2 cai nhac","2 viec")):
            if targets[0] in ("reminders.complete","tasks.complete"):
                counts[targets[0]]=2
        return cls(mode=mode,summary=str(x.get("goal",transcript))[:400],
                   apps=list(dict.fromkeys(apps)),targets=targets,target_counts=counts,
                   steps=x.get("steps",[])[:6] if isinstance(x.get("steps"),list) else [],
                   read_only=reading_only(transcript),
                   restrictions=x.get("restrictions",[])[:5] if isinstance(x.get("restrictions"),list) else [],
                   raw=x)

GOAL_SYSTEM="""You are a Vietnamese task-goal compiler for an OFFLINE assistant.
Output ONLY valid JSON:
{"mode":"act|no_op|out_of_scope","goal":"Vietnamese task summary","apps":["app_id",...],
 "target_tools":["one_or_more_installed_tools"],"read_only":false,
 "steps":["semantic step1","semantic step2"],"target_counts":{"optional_tool_name":2},"restrictions":[]}
Use installed apps and available tool names in INPUT; do not invent unavailable apps, IDs or addresses.
Pick ALL explicit goals, including second actions ("rồi nhắn", "sau đó đặt").
Choose target_tools representing final required actions (read-only request must include its read tools).
Prerequisite read tools can be added in steps, e.g. calendar.list_events before ride_hailing.book.
DO NOT confuse a negative instruction ("đừng tắt đèn, cứ để đó") with an order to turn it off: mode no_op.
If a statement is continuation of previous turns, use history to resolve the task; previous actions already applied.
Never say out_of_scope merely due uncertainty, use act and legal tools to gather info.
For request "xem tin Zalo", target is chat_ott.read_chat not chat_ott.send_message.
For request "tìm tin khoa học", target is news.search not news.save.
For "đặt xe từ công ty đi họp phụ huynh ngay", target ride_hailing.book, prerequisite calendar.list_events.
No commentary, no Markdown."""
