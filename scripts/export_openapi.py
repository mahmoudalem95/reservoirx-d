"""
Dump the OpenAPI schema for API-hub upload.

Writes two files:
  openapi.json      - OpenAPI 3.1.0, what FastAPI produces natively
  openapi-3.0.json  - a 3.0.3 downgrade, for hubs that reject 3.1

Set RXD_PUBLIC_URL to your deployed base URL before running, so the servers
block points somewhere real rather than at localhost.
"""

import copy
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.main import app  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def downgrade_to_30(schema: dict) -> dict:
    """
    Convert the 3.1-only constructs FastAPI emits into 3.0.3 equivalents.

    The two that matter: nullable fields become `anyOf: [T, {type: "null"}]` in
    3.1, and `exclusiveMinimum`/`exclusiveMaximum` change from booleans to
    numbers. Both trip up validators pinned to 3.0.
    """
    out = copy.deepcopy(schema)
    out["openapi"] = "3.0.3"

    def walk(node):
        if isinstance(node, dict):
            if "anyOf" in node and isinstance(node["anyOf"], list):
                variants = [v for v in node["anyOf"] if v != {"type": "null"}]
                if len(variants) < len(node["anyOf"]):
                    node["nullable"] = True
                    if len(variants) == 1:
                        node.pop("anyOf")
                        node.update(variants[0])
                    else:
                        node["anyOf"] = variants
            for key in ("exclusiveMinimum", "exclusiveMaximum"):
                if isinstance(node.get(key), (int, float)) and not isinstance(node[key], bool):
                    bound = "minimum" if key == "exclusiveMinimum" else "maximum"
                    node[bound] = node.pop(key)
                    node[key] = True
            if isinstance(node.get("const"), (str, int, float)):
                node["enum"] = [node.pop("const")]
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(out)
    return out


def inline_request_examples(schema: dict) -> dict:
    """
    Copy each model's example up into its requestBody.

    FastAPI puts the example on the component schema and points the requestBody
    at it with a $ref. Some API-hub consoles do not follow the ref to find it,
    so they render an empty body and every POST comes back 422 "Field required".
    Duplicating the example directly under requestBody.content covers both
    conventions; it is redundant, not wrong.
    """
    components = schema.get("components", {}).get("schemas", {})
    for item in schema.get("paths", {}).values():
        for method in ("post", "put", "patch"):
            operation = item.get(method)
            if not operation:
                continue
            content = operation.get("requestBody", {}).get("content", {}).get("application/json")
            if not content:
                continue
            ref = content.get("schema", {}).get("$ref", "")
            example = components.get(ref.split("/")[-1], {}).get("example")
            if example is not None and "example" not in content:
                content["example"] = example
                content["examples"] = {"default": {"summary": "Example request", "value": example}}
    return schema


schema = inline_request_examples(app.openapi())
(ROOT / "openapi.json").write_text(json.dumps(schema, indent=2))
(ROOT / "openapi-3.0.json").write_text(json.dumps(downgrade_to_30(schema), indent=2))

print(f"servers: {[s['url'] for s in schema.get('servers', [])]}")
print(f"wrote openapi.json      ({len(schema['paths'])} paths, OpenAPI {schema['openapi']})")
print("wrote openapi-3.0.json  (3.0.3 downgrade for stricter hubs)")
