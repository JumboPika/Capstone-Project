
#!/bin/bash

WIFI_INTERFACE="wlan0"
AP_SSID="nanoAPika"
AP_PSK="12345678"
AP_CONF="/etc/hostapd/hostapd.conf"


echo "~~~ Stopping wpa_supplicant ~~~"
# 先清理现有的连接
sudo killall wpa_supplicant
echo "~~~ Setting interface  ~~~"
sudo ip link set $WIFI_INTERFACE down
sleep 2
sudo ip link set $WIFI_INTERFACE up

# 配置 hostapd
echo "~~~ Configuring hostapd ~~~"
cat <<EOT | sudo tee $AP_CONF
interface=$WIFI_INTERFACE
ssid=$AP_SSID
hw_mode=g
channel=6
wpa=2
wpa_passphrase=$AP_PSK
wpa_key_mgmt=WPA-PSK
wpa_pairwise=TKIP
rsn_pairwise=CCMP
EOT

#
if [ ! -f "$AP_CONF" ]; then
        echo "~~~ Failed to write hostapd ~~~"
        exit 1;
fi

# 启动 hostapd 和 dnsmasq
echo "~~~ Starting hostapd ~~~"
sudo hostapd $AP_CONF &
HOSTAPD_PID=$!
sleep 5

echo "~~~ Starting dnsmasq ~~~"
sudo dnsmasq --interface=$WIFI_INTERFACE --dhcp-range=192.168.4.2,192.168.4.20,255.255.255.0,24h
DNSMASQ_PID=$!
sleep 5

#if ! pgrep -x "hostapd" > /dev/null; then
if ps -p $HOSTAPD_PID > /dev/null && ps -p $DNSMASQ_PID > /dev/null; then
        #echo "!!! hostapd failed to start !!!"
        echo "~~~ AP mode enabled. SSID: $AP_SSID ~~~"
        #exit 1
#fi
else
        echo "!!! Failed to start AP mode !!!"
#if ! pgrep -x "dnsmasq" > /dev/null; then
        #echo "!!! dnsmasq failed to start !!!"
        exit 1
fi

#echo "~~~ AP mode enabled. SSID: $AP_SSID ~~~"

# 启动 Flask 应用
echo "~~~ Starting Flask server ~~~"
cd /home/pika/Desktop/wifi || { echo "!!! Failed to change directory !!!"; exit 1; }
sudo /home/pika/archiconda3/bin/python app.py
