# Видео-инструкция «Как доехать до RE:FRAME»

Исходник: `source/IMG_7435.MP4` (2:46, 1280×720). Результат: `../brag-output/brag.mp4` (0:56, 1920×1080) и `../brag-output/brag-vertical.mp4` (0:56, 1080×1920).

## Как пересобрать

```bash
pip install opencv-python-headless numpy pillow   # один раз
video/scripts/extract_frames.sh                    # исходник -> video/work/frames/00000.jpg ... (4984 кадра)
python3 -I video/scripts/build_plates.py video/work/frames video/assets/plates   # чистые подложки карт
python3 -I video/scripts/build_routes.py                                          # маршрут, привязанный к дороге
python3 -I video/scripts/render.py video/work/frames brag-output/brag.mp4       # 16:9 (~3 мин)
python3 -I video/scripts/render.py video/work/frames brag-output/brag-vertical.mp4 --format vertical   # 9:16
```

Быстро посмотреть отдельные моменты без полного рендера (секунды выходного видео):

```bash
python3 -I video/scripts/render.py video/work/frames video/work/stills --stills 8,29.2,51.5
```

## Что где править

Всё в `video/edit.json`:

- `timemap` — `[доля, секунда исходника]`: какой момент поездки показывается на какой доле музыки (85 BPM, 1 доля = 0.706 с). Расстояние между точками задаёт ускорение.
- `steps` — список шагов справа (заголовок, подпись, на каких долях активен).
- `cards` — карточка «СЕЙЧАС» (иконка: `straight`, `left`, `left_look`, `pin`, `check`; `accent` — жёлтая, `pulse` — пульсирует).
- `turn_hud` — когда показывать жёлтую плашку «НАЛЕВО» поверх видео; `src_now` — с какой секунды исходника «СКОРО НАЛЕВО» меняется на «НАЛЕВО».
- `c_route` — синхронизация маркера на последней карте с видео: поворот (72.0 с) и прибытие (142.5 с).
- `routes` — как маршрут прижимается к дороге: `rows` — проследить дорогу по строкам карты, `seed` — точка на дороге для старта, `extra_guide` — опорные точки, если трассировку уводит (табличка на дороге, боковой съезд). После правки — `build_routes.py`, проверка в `video/work/routes_debug.jpg`.
- `maps` — координаты карты в исходном кадре; `erase_dark` / `erase_light` — что стирать (огромная стрелка, надпись «Дорога А-180»).

Подложки карт (`assets/plates/*.png`) — обычные PNG, их можно поправить в любом редакторе. Трек маркера — `assets/pin_track.json`.

Музыка — оригинальная из исходника, без ускорения (играет с 0.287 с, чтобы склейки попадали в долю), с затуханием в конце и нормализацией до −16 LUFS.
