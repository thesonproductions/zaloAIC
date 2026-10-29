"""Deterministic LifeMate V2 tests using only train JSONL and official ToolEnv APIs."""
from __future__ import annotations
import json,sys,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"datasets"/"train+public_test"
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(DATA/"train"))
from zaic_tools import ToolEnv
from lifemate_v2.controller import run_episode, verify_action,Episode
from lifemate_v2.intent import GoalFrame,pure_negative,reading_only
from lifemate_v2.retrieval import select_tools,installed_catalog

def train_case(id):
    with (DATA/"train/train.jsonl").open(encoding="utf-8") as f:
        for ln in f:
            sample=json.loads(ln)
            if sample["id"]==id:return sample
    raise KeyError(id)

CAT=json.loads((DATA/"train/train_tools_catalog.json").read_text(encoding="utf-8"))

class MockModels:
    def __init__(self,*responses):
        self.responses=list(responses)
        self.transcribe_count=0
    def transcribe(self,_):
        self.transcribe_count+=1
        raise AssertionError("No ASR when --oracle in train unit tests")
    def plan(self,content,max_tokens,system):
        if not self.responses:raise AssertionError("Unexpected extra model request")
        x=self.responses.pop(0)
        return json.dumps(x,ensure_ascii=False) if isinstance(x,dict) else x

def infer(id,*answers,max_model_calls=8):
    item=train_case(id)
    model=MockModels(*answers)
    env=ToolEnv("train",item["id"],root=DATA)
    submission,diag=run_episode(item,"train",env,model,CAT,DATA,oracle=True,
                                max_model_calls=max_model_calls,deadline_sec=100)
    return item,submission,diag,model

def gold_compare(item,output):
    # Separate evaluation-only ToolEnv for replay; not a speculative inference call.
    env=ToolEnv("train",item["id"],root=DATA)
    for call in item["label"]["tool_calls"]:
        assert env.validate(call["name"],call["arguments"])["ok"]
        env.call(call["name"],call["arguments"])
    return env.get_state_hash()==output["state_hash"]

class IntentTests(unittest.TestCase):
    def test_pure_negative(self):
        self.assertTrue(pure_negative("Em ơi, đừng tắt đèn bếp nhé, cứ để đó."))
        self.assertFalse(pure_negative("Đừng tắt đèn nhưng bật quạt cho chị."))
        self.assertFalse(pure_negative("Cho tôi biết giá xăng mới"))
        self.assertFalse(pure_negative("Sử dụng đèn trong phòng khách"))
        self.assertFalse(pure_negative("Dừng báo thức sáng mai"))
        self.assertFalse(pure_negative("Đừng quên mua quà cưới cho chị nhé"))
        self.assertFalse(pure_negative("Đừng tắt đèn nhưng mở cửa nhà"))
    def test_read_only(self):
        self.assertTrue(reading_only("Mở giúp anh tin nhắn Zalo của Mai Anh xem nhắn gì"))
        self.assertTrue(reading_only("Tìm tin gì mới về khoa học để đọc"))
        self.assertFalse(reading_only("Gửi tin nhắn cho Hương"))
        self.assertFalse(reading_only("Xem cảnh báo cửa gara, nếu mở thì khóa lại rồi nhắn cho Minh"))
        self.assertTrue(reading_only("Xem tin nhắn Zalo của Mai Anh cô ấy nhắn gì"))
    def test_retrieve_dependent(self):
        sample=train_case("train_0005")
        frame=GoalFrame(summary="book ride to parent meeting",apps=["ride_hailing"],
                        targets=["ride_hailing.book"])
        names={t["name"] for t in select_tools(sample,CAT,frame,sample["audio_text"])}
        self.assertIn("calendar.list_events",names)
        self.assertIn("ride_hailing.book",names)
    def test_multiple_same_tool_goals(self):
        frame=GoalFrame(targets=["reminders.complete"],target_counts={"reminders.complete":2})
        self.assertEqual(frame.pending([]),["reminders.complete","reminders.complete"])
        self.assertEqual(frame.pending(["reminders.complete"]),["reminders.complete"])
        self.assertEqual(frame.pending(["reminders.complete","reminders.complete"]),[])
    def test_positive_command_overrides_model_noop(self):
        item=train_case("train_1753")
        frame=GoalFrame.from_response(
            {"mode":"no_op","goal":"mua quà cưới","apps":[],"target_tools":[]},
            item["audio_text"],item["context"]["installed_apps"])
        self.assertEqual(frame.mode,"act")
    def test_app_schema_allowlist(self):
        item=train_case("train_0004")
        allowed=installed_catalog(item,CAT)
        self.assertIn("sys.ask_missing_info",allowed)
        self.assertNotIn("banking.transfer",allowed)

class ToolIntegrationTests(unittest.TestCase):
    def test_weather_condition_false_from_history_zero_calls(self):
        item,out,trace,models=infer("train_0082")
        self.assertEqual(out["tool_calls"],[])
        self.assertEqual(trace["stop_reason"],"existing_forecast_false_no_op")
        self.assertTrue(gold_compare(item,out))
        self.assertEqual(len(models.responses),0)

    def test_weather_branch_true_multi_step(self):
        frame={"mode":"act","goal":"if raining cancel dentist and sms dad",
               "apps":["weather","calendar","contacts","sms"],
               "target_tools":["calendar.cancel_event","sms.send"]}
        # Prerequisite reads and grounded cancellation are now handled
        # deterministically; the policy only supplies the final SMS proposal.
        item,out,trace,models=infer("train_0165",frame,
            {"tool":"sms.send","arguments":{"contact_id":"contact_ba","content":"Please inform dad the appointment was cancelled"}})
        self.assertEqual([x["name"] for x in out["tool_calls"]], [
            "weather.get_forecast","calendar.list_events","calendar.cancel_event",
            "contacts.search","sms.send"])
        self.assertTrue(gold_compare(item,out),trace)

    def test_alarm_relative_history_event_in_japan(self):
        frame={"mode":"act","goal":"alarm an hour before Japan conference",
               "apps":["alarm"],"target_tools":["alarm.create"]}
        item,out,trace,_=infer("train_1507",frame,
            {"tool":"alarm.create","arguments":{"time":"2026-10-18T09:00:00+07:00"}})
        self.assertEqual(out["tool_calls"][0]["arguments"],{"time":"2026-10-17T07:00:00+07:00"})
        self.assertTrue(gold_compare(item,out),trace)

    def test_two_reminders_complete_from_observed_list(self):
        frame={"mode":"act","goal":"finish two reminders robot and wallet",
               "apps":["reminders"],"target_tools":["reminders.complete"],
               "target_counts":{"reminders.complete":2}}
        item,out,trace,model=infer("train_1218",frame,
            {"tool":"reminders.list","arguments":{}})
        self.assertEqual([x["name"] for x in out["tool_calls"]],
                         ["reminders.list","reminders.complete","reminders.complete"])
        self.assertEqual({x["arguments"].get("reminder_id") for x in out["tool_calls"][1:]},
                         {"rem_513","rem_27"})
        self.assertTrue(gold_compare(item,out),trace)

    def test_reminders_list_plus_two_completions_from_goal(self):
        frame={"mode":"act","goal":"finish two reminders", "apps":["reminders"],
               "target_tools":["reminders.list","reminders.complete"]}
        item,out,trace,_=infer("train_1218",frame,{"tool":"reminders.list","arguments":{}})
        self.assertEqual(len(out["tool_calls"]),3,trace)
        self.assertTrue(gold_compare(item,out),trace)

    def test_time_grounded_reminder_tomorrow(self):
        item,out,trace,_=infer("train_0132",
            {"mode":"act","goal":"remind call customer","apps":["reminders"],
             "target_tools":["reminders.create"]},
            {"tool":"reminders.create","arguments":{"content":"gọi lại cho khách",
                "time":"2026-11-04T08:00:00+07:00"}})
        self.assertTrue(gold_compare(item,out),trace)
        self.assertEqual(out["tool_calls"][0]["arguments"]["time"],
                         "2026-11-03T20:45:00+07:00")

    def test_unsupported_album_does_not_search_podcasts(self):
        item,out,trace,m=infer("train_0562")
        self.assertEqual(out["tool_calls"],[])
        self.assertEqual(trace["stop_reason"],"unsupported_requested_capability")
        self.assertTrue(gold_compare(item,out))

    def test_positive_command_not_false_noop(self):
        x=train_case("train_1629")
        f=GoalFrame.from_response(
            {"mode":"no_op","goal":"no action","apps":[],"target_tools":[]},
            x["audio_text"],x["context"]["installed_apps"])
        self.assertEqual(f.mode,"act")

    def test_negative_subaction_not_fan_write(self):
        x=train_case("train_0349")
        f=GoalFrame.from_response(
            {"mode":"act","apps":["smart_home"],
             "target_tools":["smart_home.fan_set","smart_home.light_set",
                             "smart_home.climate_set"]},
            x["audio_text"],x["context"]["installed_apps"])
        self.assertNotIn("smart_home.fan_set",f.targets)
        self.assertIn("smart_home.light_set",f.targets)
        self.assertIn("smart_home.climate_set",f.targets)

    def test_just_thanks_no_app_call(self):
        item,out,trace,_=infer("train_0824")
        self.assertEqual(out["tool_calls"],[])
        self.assertEqual(trace["stop_reason"],"explicit_no_op")
        self.assertTrue(gold_compare(item,out))

    def test_ambiguous_recipient_asks_info_after_independent_reminder(self):
        frame={"mode":"act","goal":"set reminder and SMS Minh",
               "apps":["reminders","sms","contacts"],
               "target_tools":["reminders.create","sms.send"]}
        item,out,trace,_=infer("train_0314",frame,
            {"tool":"reminders.create","arguments":{
                "content":"đổ xăng","time":"2026-11-12T09:00:00+07:00"}})
        self.assertEqual([t["name"] for t in out["tool_calls"]],
                         ["contacts.search","reminders.create","sys.ask_missing_info"])
        self.assertEqual(out["tool_calls"][-1]["arguments"],
                         {"tool_name":"sms.send","params":["contact_id"]})
        self.assertTrue(gold_compare(item,out),trace)

    def test_negative_no_action(self):
        item,out,trace,m=infer("train_0038")
        self.assertEqual(out["tool_calls"],[])
        self.assertEqual(trace["stop_reason"],"explicit_no_op")
        self.assertTrue(gold_compare(item,out))
    def test_smart_home(self):
        frame={"mode":"act","goal":"set small bedroom lamp to 50%",
               "apps":["smart_home"],"target_tools":["smart_home.light_set"],
               "read_only":False}
        item,out,trace,_=infer("train_0004",frame,
            {"tool":"smart_home.get_status","arguments":{}},
            {"tool":"smart_home.light_set","arguments":{"device_id":"light_bedroom2","brightness":50}})
        self.assertEqual([x["name"] for x in out["tool_calls"]],
                         ["smart_home.get_status","smart_home.light_set"])
        self.assertTrue(gold_compare(item,out))
    def test_ride_cross_app_dependency(self):
        frame={"mode":"act","goal":"book ride immediately to parent meeting",
               "apps":["ride_hailing"],"target_tools":["ride_hailing.book"],"read_only":False}
        item,out,trace,_=infer("train_0005",frame,
            {"tool":"calendar.list_events","arguments":{}},
            {"tool":"ride_hailing.book","arguments":{
                "pickup":"Khu Công nghệ cao, TP. Thủ Đức, TP.HCM",
                "dropoff":"Trường Tiểu học Nguyễn Bỉnh Khiêm, 1 Đinh Tiên Hoàng, Quận 1, TP.HCM",
                "vehicle":"car4"}})
        self.assertTrue(gold_compare(item,out))
        self.assertEqual(len(out["tool_calls"]),2)
    def test_read_chat_requires_read(self):
        frame={"mode":"act","goal":"read Mai Anh chat",
               "apps":["chat_ott"],"target_tools":["chat_ott.read_chat"],"read_only":True}
        item,out,trace,_=infer("train_0692",frame,
            {"tool":"chat_ott.list_chats","arguments":{}},
            {"tool":"chat_ott.read_chat","arguments":{"chat_id":"chat_01"}})
        self.assertTrue(gold_compare(item,out))
        self.assertEqual([x["name"] for x in out["tool_calls"]],
                         ["chat_ott.list_chats","chat_ott.read_chat"])
    def test_unrequested_news_write_never_called(self):
        frame={"mode":"act","goal":"find science news","apps":["news"],
               "target_tools":["news.search"],"read_only":True}
        item,out,trace,_=infer("train_0866",frame,
            {"tool":"news.search","arguments":{"query":"khoa học"}})
        self.assertTrue(gold_compare(item,out))
        self.assertEqual(len(out["tool_calls"]),1)
    def test_history_id_grounding(self):
        frame={"mode":"act","goal":"enroll in beginner first aid",
               "apps":["learning"],"target_tools":["learning.enroll"],"read_only":False}
        item,out,trace,_=infer("train_0630",frame,
             {"tool":"learning.enroll","arguments":{"course_id":"course_04"}})
        self.assertTrue(gold_compare(item,out))
        self.assertEqual([c["name"] for c in out["tool_calls"]],["learning.enroll"])
    def test_hallucinated_ref_blocked_before_execution(self):
        frame={"mode":"act","goal":"set room light brightness",
               "apps":["smart_home"],"target_tools":["smart_home.light_set"]}
        # Request unknown ID before any get_status: must NEVER call env with it.
        item,out,trace,_=infer("train_0004",frame,
             *[{"tool":"smart_home.light_set","arguments":{"device_id":"invented_999","brightness":50}}]*3)
        self.assertEqual(out["tool_calls"],[])
        self.assertTrue(all("unobserved x-ref" in v.get("error","") for v in trace["violations"]))
    def test_read_only_verifier_blocks_news_save(self):
        item=train_case("train_0866")
        frame=GoalFrame(mode="act",summary=item["audio_text"],apps=["news"],
                        targets=["news.search"],read_only=True)
        episode=Episode(frame,item["audio_text"],[])
        sch=select_tools(item,CAT,frame,item["audio_text"])
        ok,reason=verify_action("news.save",{"article_id":"unknown"},episode,
                                installed_catalog(item,CAT),sch,item)
        self.assertFalse(ok)

if __name__=="__main__":
    unittest.main(verbosity=2)
