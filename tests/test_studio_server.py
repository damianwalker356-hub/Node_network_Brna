import json
import threading
import urllib.error
import urllib.request

import pytest

from brna.server import create_server


@pytest.fixture
def running_server(tmp_path):
    server = create_server(port=0, data_directory=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def _request(url, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{url}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=3) as response:
        return response.status, json.loads(response.read())


@pytest.mark.req("R-STUDIO-02")
def test_server_serves_studio_and_reports_model_status(running_server):
    assert running_server.startswith("http://127.0.0.1:")
    with urllib.request.urlopen(f"{running_server}/", timeout=3) as response:
        assert response.status == 200
        assert b"BRNA Studio" in response.read()

    status, health = _request(running_server, "/api/health")
    assert status == 200
    assert health["service"] == "brna-studio"
    assert health["model"]["trained"] is False


@pytest.mark.req("R-STUDIO-02")
def test_server_teaches_remembers_and_trains_local_model(running_server):
    status, result = _request(
        running_server,
        "/api/teach",
        {"prompt": "What is BRNA?", "response": "A message-passing network."},
    )
    assert status == 200
    assert result["conversation_examples"] == 1

    status, answer = _request(running_server, "/api/chat", {"prompt": "What is BRNA?"})
    assert status == 200
    assert answer == {
        "response": "A message-passing network.",
        "source": "taught-example",
        "similarity": 1.0,
    }

    status, training = _request(
        running_server,
        "/api/train",
        {"source": "examples", "epochs": 1, "sequence_length": 32},
    )
    assert status == 200
    assert training["trained"] is True
    assert training["training_characters"] > 0

    status, health = _request(running_server, "/api/health")
    assert status == 200
    assert health["model"]["trained"] is True


@pytest.mark.req("R-STUDIO-02")
def test_server_rejects_invalid_json_and_unknown_routes(running_server):
    request = urllib.request.Request(
        f"{running_server}/api/teach",
        data=b"{",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=3)
    assert error.value.code == 400
    assert "valid UTF-8 JSON" in error.value.read().decode("utf-8")

    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(f"{running_server}/api/missing", timeout=3)
    assert error.value.code == 404
