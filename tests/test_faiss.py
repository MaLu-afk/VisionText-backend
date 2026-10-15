import asyncio

import numpy as np

from app.config import settings
from app.services.faiss import FAISSService


def make_service(ids):
    """Índice con un vector one-hot por imagen: la imagen i solo es similar a sí misma"""
    service = FAISSService()
    service.initialize_index(len(ids))
    for position, image_id in enumerate(ids):
        asyncio.run(service.add_embedding(image_id, np.eye(len(ids), dtype="float32")[position]))
    return service


def top_result(service, vector):
    results = asyncio.run(service.search(vector, k=1, threshold=0.5))
    return results[0][0] if results else None


def test_remove_keeps_ids_aligned_with_vectors():
    ids = ["a", "b", "c", "d"]
    service = make_service(ids)

    assert asyncio.run(service.remove_embedding("b"))

    assert service.get_count() == 3
    assert service.image_ids == ["a", "c", "d"]
    vectors = np.eye(4, dtype="float32")
    # Cada vector restante debe seguir devolviendo su propia imagen
    assert top_result(service, vectors[0]) == "a"
    assert top_result(service, vectors[2]) == "c"
    assert top_result(service, vectors[3]) == "d"
    # El vector borrado ya no aparece
    assert top_result(service, vectors[1]) is None


def test_remove_unknown_id_is_noop():
    service = make_service(["a", "b"])

    assert not asyncio.run(service.remove_embedding("zzz"))
    assert service.get_count() == 2


def test_load_rejects_out_of_sync_index(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "faiss_index_path", str(tmp_path / "faiss_index.bin"))
    service = make_service(["a", "b", "c"])
    # Simula el índice que dejaba la versión anterior: ID borrado, vector no
    service.image_ids.remove("b")
    asyncio.run(service.save_index())

    reloaded = FAISSService()

    assert reloaded.load_index() is False
    assert reloaded.index is None


def test_load_accepts_consistent_index(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "faiss_index_path", str(tmp_path / "faiss_index.bin"))
    asyncio.run(make_service(["a", "b"]).save_index())

    reloaded = FAISSService()

    assert reloaded.load_index() is True
    assert reloaded.image_ids == ["a", "b"]
