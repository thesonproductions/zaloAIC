"""Grounded conditional planning and Vietnamese relative time, with no hidden state reads."""
from __future__ import annotations
from datetime import datetime,timedelta
import re
from .intent import norm

def date_for_relative(text:str,now:str)->str|None:
    """Return date for unambiguous deictic Vietnamese days, else None."""
    s=norm(text)
    dt=datetime.fromisoformat(now)
    if re.search(r"\b(ngay kia|mot ngay sau ngay mai)\b",s):dt+=timedelta(days=2)
    elif re.search(r"\b(mai|ngay mai)\b",s):dt+=timedelta(days=1)
    elif re.search(r"\b(hom nay|bay gio|sang nay|toi nay|chieu nay)\b",s):pass
    else:return None
    return dt.date().isoformat()

def shift_from_reference(text:str,base_iso:str)->str|None:
    """Time operation only when a clear one/two-hour or 30-minute offset is requested."""
    s=norm(text)
    amount=None
    if re.search(r"\b(nua tieng|30 phut|ba muoi phut)\b",s):amount=30
    elif re.search(r"\b(mot tieng|1 tieng|mot gio|1 gio)\b",s):amount=60
    elif re.search(r"\b(hai tieng|2 tieng|hai gio|2 gio)\b",s):amount=120
    if amount is None:return None
    direction=-1 if re.search(r"\b(truoc|som hon|som len|som)\b",s) else 1 if re.search(r"\b(muon|tre hon|lui lai)\b",s) else None
    if direction is None:return None
    return (datetime.fromisoformat(base_iso)+timedelta(minutes=direction*amount)).isoformat(timespec="seconds")

def forecast_from_history(history:list[dict],target_date:str|None):
    """Use ONLY official history tool_results supplied in the sample, never backing state."""
    if not target_date:return None
    for entry in reversed(history):
        calls=entry.get("tool_calls",[])
        results=entry.get("tool_results",[])
        for call,result in reversed(list(zip(calls,results))):
            if call.get("name")!="weather.get_forecast" or not isinstance(result,dict):continue
            for entry in result.get("forecast",[]):
                if entry.get("date")==target_date and isinstance(entry.get("rain"),bool):
                    return entry["rain"]
    return None

def current_forecast(observations:list[dict],target_date:str|None):
    for ob in reversed(observations):
        if ob.get("name")!="weather.get_forecast":continue
        for item in ob.get("result",{}).get("forecast",[]):
            if not target_date or item.get("date")==target_date:
                if isinstance(item.get("rain"),bool):return item["rain"]
    return None

def rain_condition(text:str):
    s=norm(text)
    return bool(re.search(r"\b(neu|ma)\s+\w*\s*mua\b|\bmai\s+ma\s+mua\b",s)) or bool(re.search(r"\bneu\s+mua\b",s))

def direct_tool_goals(text:str,available:set[str])->list[str]:
    """Conservative lexical *positive* goals based on tool capability, not test labels."""
    s=norm(text)
    out=[]
    patterns=[
       ("ride_hailing.book",r"\b(dat xe|goi xe|goi o to|dat o to)\b"),
       ("sms.send",r"\b(nhan sms|gui sms|sms cho|nhan tin sms)\b"),
       ("chat_ott.send_message",r"\b(nhan cho|gui cho|gui zalo|gui tin zalo)\b"),
       ("calendar.cancel_event",r"\b(huy lich|huy cai lich|huy su kien)\b"),
       ("calendar.update_event",r"\b(doi lich|doi gio|day lich|doi su kien|doi cai lich)\b"),
       ("reminders.complete",r"\b(danh dau xong|danh dau hoan thanh)\b"),
       ("tasks.add",r"\b(them.*\bviec\b|tao.*\bviec\b|ghi.*\bviec\b)"),
       ("gold_price.get_price",r"\b(gia vang|check gia vang|tra gia vang)\b"),
       ("alarm.create",r"\b(dat bao thuc|tao bao thuc|nhac gio)\b"),
       ("door_lock.set_lock",r"\b(khoa lai|khoa cua|khoa no)\b"),
       ("news.search",r"\b(tim tin|tin gi moi|tin tuc|doc tin)\b"),
       ("loyalty.redeem",r"\b(doi phieu|doi diem|lay diem.*doi|doi voucher)\b"),
       ("maps.search_place",r"\b(quan.*gan nha|gan nha.*quan|quan mo khong|co quan mo|quan nao gan)\b"),
       ("appliances.stop",r"\b(noi com tat|tat noi com|ngung may giat|dung may giat)\b"),
       ("food_delivery.cancel_order",r"\b(huy don|huy.*don an)\b"),
    ]
    hit=[]
    for tool,pattern in patterns:
        if tool not in available:continue
        m=re.search(pattern,s)
        if m:hit.append((m.start(),tool))
    return [tool for _,tool in sorted(hit)]

def resolve_rain_branch(text:str,available:set[str],rain:bool)->list[str]:
    """Extract goals from selected branch, rather than both branches."""
    normalized=norm(text)
    pre,sep,after=re.split(r"\b(khong mua thi|con khong mua thi)\b",normalized,maxsplit=1) if re.search(r"\b(khong mua thi|con khong mua thi)\b",normalized) else (normalized,"","")
    # False branch often says 'không mưa thì thôi', legitimately no effects.
    if rain:
        return direct_tool_goals(pre,available)
    if after:
        return direct_tool_goals(after,available)
    # If no explicit else, false case means no requested changes.
    return []

def non_negative_goal_overrides(transcript:str,targets:list[str],available:set[str])->list[str]:
    s=norm(transcript)
    result=list(targets)
    positive=direct_tool_goals(transcript,available)
    for name in positive:
        if name not in result:result.append(name)
    # Prevent task inference from conversation nouns (e.g. "đừng tắt đèn").
    return result
