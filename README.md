# Nuerosemantic test task

Для реализации тестового задания использовался следующий стек технологий:
- Веб сервер FastAPI
- Инференс сетей Triton Inference Server
- Python 3.10
## Сетки, которые используются 
- [Yolov8n][df2] - для нахождения лиц в кадре
- [MiVOLO][df1] - для определения пола и возраста

[df1]: <https://github.com/WildChlamydia/MiVOLO>
[df2]: <https://github.com/derronqi/yolov8-face>
Все сетки сконвертированны в формат onnx, что упрощает их запуск на различных устройства

## Installation

Для запуска необходимо установить Docker и Docker compose


```sh
склонить репозиторий
docker compose up
```

## Сваггер доступен по [ссылке][swagger]
[swagger]: <http://localhost:5000/docs>