"""Filtro de menus por OCR: señal NEGATIVA, no positiva.

En gameplay el frame con MAS texto es el lobby, que es lo menos publicable.
Medido en un video real de Warzone: lobby 35 textos, gameplay 14-16.
"""

from app.pipeline.menu_filter import (
    MENU_TEXT_COUNT,
    filter_menu_candidates,
    looks_like_cutscene,
    looks_like_menu,
)
from app.providers.base import ClipCandidate


def test_detecta_lobby_por_terminos_reales():
    # Textos reales leidos del lobby de Warzone en el video de prueba
    textos = [
        "votacion en curso:0",
        "pase de batalla",
        "reservar sala",
        "operadores",
        "tienda",
        "jiggylivin",
    ]
    menu, why = looks_like_menu(textos)
    assert menu
    assert "menu" in why


def test_gameplay_no_es_menu():
    # Textos reales de frames de partida
    assert looks_like_menu(["301", "mk351sr", "calle principal", "vigilancia"])[0] is False


def test_scoreboard_es_interfaz_no_publicable():
    rejected, why = looks_like_menu(["scoreboard", "equipo", "puntos"])

    assert rejected is True
    assert "menu" in why


def test_pantalla_te_elimino_es_interfaz_no_publicable():
    rejected, why = looks_like_menu(["matarreyes", "teelimino:", "eliminacion"])

    assert rejected is True
    assert "pantalla de derrota" in why


def test_ocr_que_pierde_te_pero_lee_elimino_sigue_detectando_derrota():
    rejected, why = looks_like_menu(["atacantes 1", "elimino", "uav"])

    assert rejected is True
    assert "pantalla de derrota" in why


def test_eliminacion_conseguida_no_se_confunde_con_derrota():
    assert looks_like_menu(["matarreyes", "eliminacion", "+100"])[0] is False


def test_overlay_del_streamer_no_marca_menu():
    """El overlay de marca sale en los 2200 fotogramas: no puede disparar el filtro."""
    textos = ["tiktok", "instagram", "twitch", "kick", "@kerwil", "linkbio", "datos:on", "gpu:63"]
    assert looks_like_menu(textos)[0] is False


def test_densidad_alta_marca_menu_sin_terminos_conocidos():
    """Un juego en otro idioma no matchea terminos, pero un menu igual satura de texto."""
    textos = [f"palabra{i}" for i in range(MENU_TEXT_COUNT + 3)]
    menu, why = looks_like_menu(textos)
    assert menu
    assert "densidad" in why


def test_sin_texto_no_es_menu():
    assert looks_like_menu([])[0] is False


# --- cutscene/dialogo de modo historia mezclado en un montaje "Highlights" -----------
# Textos reales leidos por RapidOCR de frames de Mortal Kombat 1 (job real, F38/F39):
# una escena de dialogo entre 2 personajes quedo seleccionada como beat narrativo en
# medio de un montaje de jugadas de partidas distintas, sin sentido para el usuario.


def test_dialogo_de_cutscene_sin_hud_se_detecta():
    # RapidOCR fundio el subtitulo completo en un solo bloque sin espacios.
    textos = ["en vivo", "@kerwil", "harumiestanferozcomoel."]
    cutscene, why = looks_like_cutscene(textos)
    assert cutscene
    assert "dialogo" in why
    assert looks_like_menu(textos) == (True, why)


def test_fatal_blow_con_hud_no_es_cutscene():
    assert looks_like_cutscene(["en vivo", "@kerwil", "53", "fatal dlgw"]) == (False, "")


def test_hud_de_vida_fraccionario_nunca_es_cutscene_aunque_haya_texto_largo():
    # Un subtitulo/etiqueta larga en pantalla, pero con el HUD de combate presente,
    # sigue siendo gameplay real.
    textos = ["4.49/1000", "una frase bastante larga de anuncio del combo activo"]
    assert looks_like_cutscene(textos) == (False, "")


def test_overlay_del_streamer_solo_no_es_cutscene():
    # Frames sin HUD legible (motion blur, efectos) pero tambien sin subtitulo largo:
    # no debe marcarse cutscene solo por ausencia de HUD.
    assert looks_like_cutscene(["en vivo", "@kerwil"]) == (False, "")


def test_termino_de_menu_largo_no_se_confunde_con_cutscene():
    assert looks_like_cutscene(["partida privada", "buscar partida"]) == (False, "")


def test_filtro_nunca_deja_el_pool_vacio(monkeypatch):
    """Un pool vacio rompe el job; un menu colado no. Mismo criterio que SponsorBlock."""
    import app.pipeline.menu_filter as mod

    monkeypatch.setattr(mod._Engine, "get", classmethod(lambda cls: object()))
    monkeypatch.setattr(mod, "is_menu_frame", lambda p: (True, "todo es menu"))

    def fake_run(cmd, **kwargs):
        from pathlib import Path

        Path(cmd[-2]).write_bytes(b"frame")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    cands = [ClipCandidate(start=0, end=30, score=0.5), ClipCandidate(start=60, end=90, score=0.5)]
    out = filter_menu_candidates(cands, "fake.mp4")
    assert out == cands


def test_filtro_revisa_apertura_y_descarta_menu_aunque_el_centro_sea_gameplay(monkeypatch):
    import app.pipeline.menu_filter as mod

    monkeypatch.setattr(mod._Engine, "get", classmethod(lambda cls: object()))
    sampled: list[float] = []

    def fake_run(cmd, **kwargs):
        from pathlib import Path

        sampled.append(float(cmd[cmd.index("-ss") + 1]))
        Path(cmd[-2]).write_bytes(b"frame")

    def fake_is_menu(path):
        return ("c0-s0" in path, "texto de menu: armamento")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod, "is_menu_frame", fake_is_menu)

    menu = ClipCandidate(start=10, end=90, score=0.7)
    gameplay = ClipCandidate(start=100, end=180, score=0.6)
    out = filter_menu_candidates([menu, gameplay], "fake.mp4")

    assert out == [gameplay]
    assert sampled[0] == 15.0
    assert 50.0 not in sampled, "no debe pagar OCR central si la apertura ya es menu"


def test_filtro_es_noop_sin_video_o_sin_ocr():
    cands = [ClipCandidate(start=0, end=30, score=0.5)]
    assert filter_menu_candidates(cands, None) == cands
    assert filter_menu_candidates([], "fake.mp4") == []
