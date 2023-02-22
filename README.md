## setup

1. Install dependencies.

    ```bash
    pip install -r requirements.txt
    git clone https://github.com/lllyasviel/ControlNet.git
    ```

2. Then you can generate images. The kwargs parameter is optional.

    ```bash
    python main.py --image-uri anime_girl.png --model canny --prompt "anime girl" --steps 50 --kwargs '{"low_threshold" : 150, "high_threshold":200}'
    ```