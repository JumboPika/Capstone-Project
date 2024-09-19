from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ssid = request.form['ssid']
        password = request.form['password']

        # 更新 wpa_supplicant 配置
        update_wpa_supplicant(ssid, password)

        return jsonify({"status": "success", "message": "WiFi configuration updated. You can now switch to client mode."})

    return render_template('index.html')

def update_wpa_supplicant(ssid, password):
    # wpa_supplicant 配置文件路徑
    wpa_supplicant_conf = '/etc/wpa_supplicant/wpa_supplicant.conf'

    # 讀取現有配置
    with open(wpa_supplicant_conf, 'r') as f:
        lines = f.readlines()

    # 準備新網絡條目
    new_entry = f'\nnetwork={{\n ssid="{ssid}"\n psk="{password}"\n}}\n'

    # 創建新的配置內容，覆蓋現有的 SSID 條目（如果存在）
    new_lines = []
    in_network_block = False
    network_block = ""

    for line in lines:
        if line.strip().startswith('network={'):
            in_network_block = True
            network_block = line
        elif in_network_block:
            network_block += line
            if line.strip() == '}':
                in_network_block = False
                # 檢查這是否是需要更新的網絡條目
                if f'ssid="{ssid}"' in network_block:
                    continue  # 跳過此條目，因為它將被替換
                else:
                    new_lines.append(network_block)
        else:
            new_lines.append(line)

    # 添加新條目
    new_lines.append(new_entry)

    # 將新的配置寫回文件
    with open(wpa_supplicant_conf, 'w') as f:
        f.writelines(new_lines)

    # 重新加載 wpa_supplicant 以應用更改
    os.system('sudo wpa_cli reconfigure')

@app.route('/switch_to_client', methods=['POST'])
def switch_to_client():
    try:
        # 執行 switch_to_client.sh 腳本
        os.system('sudo /usr/local/bin/switch_to_client.sh')
        return jsonify({"status": "success", "message": "Switched to client mode. Attempting to connect to WiFi."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
