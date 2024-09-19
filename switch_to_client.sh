#!/bin/bash

# 介面名稱
WIFI_INTERFACE="wlan0"

# wpa_supplicant 配置文件路徑
WPA_CONF="/etc/wpa_supplicant/wpa_supplicant.conf"

# 停止 AP 模式相關服務
echo "Stopping AP mode services (hostapd, dnsmasq) ---"
sudo systemctl stop hostapd
sudo systemctl stop dnsmasq

# 先清理現有的連接
sudo killall wpa_supplicant
sudo ip link set $WIFI_INTERFACE down
sudo ip link set $WIFI_INTERFACE up

echo "Preparing to connect to WIFI ---"

# 嘗試連接到 WiFi
sudo timeout 10s wpa_supplicant -B -i $WIFI_INTERFACE -c $WPA_CONF
sudo timeout 10s dhclient $WIFI_INTERFACE

# 等待 10 秒讓連接完成
# sleep 10

echo "Checking the connection ---"

# 嘗試 ping Google 確認網路連線
if ping -c 3 -W 5 google.com &>/dev/null; then
    echo "Connected to WIFI successfully ---"
    exit 0
else
    echo "!!! Failed to connect to WIFI !!!"
    echo "Switching to AP mode..."
    source /usr/local/bin/switch_to_ap.sh
    exit 1
fi
