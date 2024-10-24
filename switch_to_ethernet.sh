#!/bin/bash

# 介面名稱
WIFI_INTERFACE="wlan0"
ETH_INTERFACE="eth0"

echo "Switching to Ethernet connection ---"

# 停止 WiFi 連接
echo "Stopping WIFI connection ---"
sudo killall wpa_supplicant
sudo ip link set $WIFI_INTERFACE down

# 啟用有線網卡
echo "Bringing up Ethernet interface ---"
sudo ip link set $ETH_INTERFACE up

# 獲取 IP 地址
echo "Obtaining IP address via DHCP ---"
sudo dhclient $ETH_INTERFACE

# 確認是否連接到網路
if ping -c 3 -W 10 google.com &>/dev/null; then
    echo "Connected to Ethernet successfully ---"
    exit 0
else
    echo "!!! Failed to connect to Ethernet !!!"
    echo "Check the Ethernet connection or try again."
    exit 1
fi
