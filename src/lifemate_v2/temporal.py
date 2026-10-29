"""Vietnamese deictic time normalization grounded in sample context, not wall clock."""
from __future__ import annotations
from datetime import datetime,timedelta
import re
from .intent import norm

HOURS={"mot":1,"hai":2,"ba":3,"bon":4,"tu":4,"nam":5,"sau":6,
       "bay":7,"tam":8,"chin":9,"muoi":10,"muoi mot":11,"muoi hai":12}
def hour_token(token:str)->int|None:
    token=token.strip()
    if token.isdigit():return int(token)
    return HOURS.get(token)

def explicit_clock(text:str,reference_iso:str)->str|None:
    """Parse explicit Vietnamese '8 giờ sáng mai', '9 giờ kém 15 tối mai'.
    Avoid guessing when date/time is not explicitly said; never read hidden state.
    """
    s=norm(text)
    clock=re.search(r"\b(\d{1,2}|muoi hai|muoi mot|muoi|chin|tam|bay|sau|nam|tu|bon|ba|hai|mot)\s+gio\s*(?:(kem\s+(?:\d{1,2}|muoi lam)|ruoi)|)?",s)
    if not clock:return None
    h=hour_token(clock.group(1))
    if h is None or h>23:return None
    m=0
    if clock.group(2):
        offset=clock.group(2)
        if offset=="ruoi":m=30
        elif offset.startswith("kem "):
            amount=offset[4:]
            mm=15 if amount=="muoi lam" else int(amount)
            if mm>=60:return None
            # In '9 giờ kém 15 tối', subtract from 9pm, not from 9am.
            m=-mm
    after=s[clock.end():]
    before=s[:clock.start()]
    # Use the original accented text for day-part; unaccented "toi" could
    # otherwise mean "tôi" (I) rather than "tối" (evening).
    lower=text.casefold()
    window=lower[clock.end():clock.end()+36]+" "+lower[max(0,clock.start()-26):clock.start()]
    # Explicit day is mandatory, otherwise caller must not invent a calendar date.
    ref=datetime.fromisoformat(reference_iso)
    if re.search(r"\b(ngay kia|mot ngay sau ngay mai)\b",s):
        target=ref+timedelta(days=2)
    elif re.search(r"\b(mai|ngay mai)\b",s):
        target=ref+timedelta(days=1)
    elif re.search(r"\b(hom nay|nay|toi nay|chieu nay|sang nay)\b",window):
        target=ref
    else:return None
    if re.search(r"\b(tối|đêm|chiều)\b",window) and h<=11:
        h+=12
    elif re.search(r"\b(sáng)\b",window):
        if h==12:h=0
    if h>23:return None
    dt=target.replace(hour=h,minute=0,second=0,microsecond=0)+timedelta(minutes=m)
    return dt.isoformat(timespec="seconds")
