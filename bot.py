import os
import sys
import time
import shutil
import subprocess
import urllib.request
import json

# ============================================================
# CONFIG
# ============================================================
BOT_TOKEN = "7903173759:AAFjjDIe6P0p5HqDFurO1qXBevrNzDuoXJY"
CHAT_ID = 6424066420

STREAMLIT_PORT = 8501

if not BOT_TOKEN or not CHAT_ID:
    print("ERROR: BOT_TOKEN and CHAT_ID are required.")
    print("Set them with:")
    print('export BOT_TOKEN="YOUR_BOT_TOKEN"')
    print('export CHAT_ID="YOUR_CHAT_ID"')
    sys.exit(1)


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    data = urllib.parse.urlencode({
        "chat_id": CHAT_ID,
        "text": message
    }).encode()

    try:
        urllib.request.urlopen(url, data=data, timeout=15)
        print("Telegram message sent.")
    except Exception as e:
        print("Telegram error:", e)


# ============================================================
# RUN COMMAND
# ============================================================

def run_command(command):
    print("\n================================")
    print("RUNNING:", command)
    print("================================\n")

    result = subprocess.run(
        command,
        shell=True
    )

    return result.returncode


# ============================================================
# CHECK CLOUDFLARED
# ============================================================

def install_cloudflared():

    if shutil.which("cloudflared"):
        print("cloudflared already installed.")
        return True

    print("cloudflared not found.")

    print("Downloading cloudflared...")

    url = (
        "https://github.com/cloudflare/cloudflared/"
        "releases/latest/download/"
        "cloudflared-linux-amd64"
    )

    try:
        urllib.request.urlretrieve(
            url,
            "/usr/local/bin/cloudflared"
        )

        os.chmod(
            "/usr/local/bin/cloudflared",
            0o755
        )

        print("cloudflared installed.")
        return True

    except Exception as e:
        print("Could not install cloudflared:", e)
        return False


# ============================================================
# MAIN
# ============================================================

def main():

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    os.chdir(base_dir)

    # --------------------------------------------------------
    # 1. TRAIN MODEL
    # --------------------------------------------------------

    send_telegram(
        "🧠 Starting ML training...\n"
        "Running train.py"
    )

    code = run_command(
        f"{sys.executable} train.py"
    )

    if code != 0:

        send_telegram(
            "❌ Training failed.\n"
            f"Exit code: {code}"
        )

        sys.exit(code)

    send_telegram(
        "✅ Training completed successfully.\n"
        "PKL models generated."
    )

    # --------------------------------------------------------
    # 2. INSTALL CLOUDFLARED
    # --------------------------------------------------------

    if not install_cloudflared():

        send_telegram(
            "❌ Could not install cloudflared."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # 3. START STREAMLIT
    # --------------------------------------------------------

    send_telegram(
        "🚀 Starting Streamlit..."
    )

    streamlit_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.address",
            "0.0.0.0",
            "--server.port",
            str(STREAMLIT_PORT),
            "--server.headless",
            "true"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # Give Streamlit time to start
    time.sleep(8)

    if streamlit_process.poll() is not None:

        send_telegram(
            "❌ Streamlit failed to start."
        )

        sys.exit(1)

    print("Streamlit started.")


    # --------------------------------------------------------
    # 4. START CLOUDFLARE TUNNEL
    # --------------------------------------------------------

    send_telegram(
        "🌐 Creating public Streamlit URL..."
    )

    tunnel_process = subprocess.Popen(
        [
            "cloudflared",
            "tunnel",
            "--url",
            f"http://127.0.0.1:{STREAMLIT_PORT}"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    public_url = None

    start_time = time.time()

    while time.time() - start_time < 30:

        line = tunnel_process.stdout.readline()

        if not line:
            time.sleep(0.2)
            continue

        print(line, end="")

        # Cloudflare Quick Tunnel URL
        if "trycloudflare.com" in line:

            parts = line.split()

            for part in parts:

                if "https://" in part and "trycloudflare.com" in part:

                    public_url = part.strip()

                    break

        if public_url:
            break


    # --------------------------------------------------------
    # 5. SEND URL TO TELEGRAM
    # --------------------------------------------------------

    if public_url:

        message = (
            "🎉 STREAMLIT APP IS ONLINE!\n\n"
            f"🔗 {public_url}\n\n"
            "🧠 ML models loaded.\n"
            "🚀 Streamlit running."
        )

        send_telegram(message)

        print("\nPUBLIC URL:")
        print(public_url)

    else:

        send_telegram(
            "⚠️ Streamlit started, but I couldn't "
            "detect the Cloudflare public URL."
        )

        print(
            "Could not detect public URL."
        )


    # --------------------------------------------------------
    # 6. KEEP EVERYTHING RUNNING
    # --------------------------------------------------------

    print("\n================================")
    print("SYSTEM RUNNING")
    print("================================")

    try:

        while True:

            # Streamlit crashed?
            if streamlit_process.poll() is not None:

                print("Streamlit stopped.")

                send_telegram(
                    "⚠️ Streamlit process stopped."
                )

                break

            # Tunnel crashed?
            if tunnel_process.poll() is not None:

                print("Cloudflare tunnel stopped.")

                send_telegram(
                    "⚠️ Cloudflare tunnel stopped."
                )

                break

            time.sleep(5)

    except KeyboardInterrupt:

        print("\nStopping...")

        streamlit_process.terminate()
        tunnel_process.terminate()


if __name__ == "__main__":
    main()