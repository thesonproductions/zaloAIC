import unittest
import sys
from pathlib import Path

from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline_agent" / "adk"))
from zaic_adk import SimulatorTool, _adk_schema


class FakeEnv:
    def __init__(self):
        self.calls = []
        self.state = {"power": "on"}

    def call(self, name, arguments):
        self.calls.append({"name": name, "arguments": arguments})
        if name == "smart_home.light_set":
            self.state["power"] = arguments["power"]
        if name == "smart_home.get_status":
            return {"ok": True, **self.state}
        return {"ok": True}


class AdkBenchmarkTests(unittest.IsolatedAsyncioTestCase):
    def test_catalog_schema_maps_to_genai_schema(self):
        schema = _adk_schema({
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["on", "off"]},
                "items": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["mode"],
            "x-one-of": [["contact_id"], ["phone_number"]],
        })

        self.assertEqual(schema.type, types.Type.OBJECT)
        self.assertEqual(schema.properties["mode"].enum, ["on", "off"])
        self.assertEqual(schema.properties["items"].items.type, types.Type.INTEGER)
        self.assertEqual(schema.required, ["mode"])

    async def test_adk_tools_share_one_mutable_zaic_session(self):
        env = FakeEnv()
        write_tool = SimulatorTool({
            "name": "smart_home.light_set",
            "description": "Set a light state.",
            "parameters": {
                "type": "object",
                "properties": {"power": {"type": "string", "enum": ["on", "off"]}},
                "required": ["power"],
            },
        }, env)
        read_tool = SimulatorTool({
            "name": "smart_home.get_status",
            "description": "Read the current state.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }, env)

        declaration = write_tool._get_declaration()
        write_result = await write_tool.run_async(
            args={"power": "off"}, tool_context=None
        )
        read_result = await read_tool.run_async(args={}, tool_context=None)

        self.assertNotEqual(declaration.name, "smart_home.light_set")
        self.assertIn("smart_home.light_set", declaration.description)
        self.assertEqual(write_result, {"ok": True})
        self.assertEqual(read_result, {"ok": True, "power": "off"})
        self.assertEqual(env.state["power"], "off")
        self.assertEqual(
            [call["name"] for call in env.calls],
            ["smart_home.light_set", "smart_home.get_status"],
        )


if __name__ == "__main__":
    unittest.main()
