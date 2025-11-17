package com.example.cybersafeconnect

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.wifi.ScanResult
import android.net.wifi.WifiManager
import android.os.Handler
import android.os.Looper
import android.widget.Toast
import androidx.core.content.ContextCompat

class WifiReceiver : BroadcastReceiver() {

    // Short helper: run toast on main thread
    private fun showToast(context: Context, text: String) {
        Handler(Looper.getMainLooper()).post {
            Toast.makeText(context.applicationContext, text, Toast.LENGTH_LONG).show()
        }
    }

    override fun onReceive(context: Context, intent: Intent?) {
        try {
            val connManager = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
            val network = connManager.activeNetwork ?: return
            val caps = connManager.getNetworkCapabilities(network) ?: return

            // Only proceed on Wi-Fi transport
            if (!caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI)) return

            val wifiManager = context.applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
            val info = wifiManager.connectionInfo
            val rawSsid = info.ssid ?: ""
            val ssid = rawSsid.replace("\"", "")
            val bssid = info.bssid // may be null on some devices

            // Need Location permission to read scanResults
            if (ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION)
                != PackageManager.PERMISSION_GRANTED) {
                // Inform user to grant permission
                showToast(context, "Please grant Location permission for Wi-Fi security checks.")
                NotificationHelper(context).createNotification(
                    "Permission Required",
                    "Allow Location access so CyberSafe Connect can check Wi-Fi security."
                )
                return
            }

            // Trigger scan for freshest results (may be rate-limited)
            try { wifiManager.startScan() } catch (_: Exception) {}

            // Small delay (optional) to allow scan to refresh; many devices return last known results immediately.
            val scanResults: List<ScanResult> = wifiManager.scanResults

            // Find the matching AP by BSSID first, then SSID
            val matched = scanResults.firstOrNull { it.BSSID.equals(bssid, ignoreCase = true) }
                ?: scanResults.firstOrNull { it.SSID == ssid }

            val capsStr = matched?.capabilities ?: ""

            // Determine whether the AP is open (no WPA/WEP/WPA2/WPA3 in capabilities)
            val isOpen = !(capsStr.contains("WPA", ignoreCase = true)
                    || capsStr.contains("WEP", ignoreCase = true)
                    || capsStr.contains("WPA2", ignoreCase = true)
                    || capsStr.contains("WPA3", ignoreCase = true))

            if (isOpen) {
                // Insecure / open network found
                val title = "⚠️ Insecure Wi-Fi Detected"
                val msg = if (ssid.isNotBlank()) "$ssid appears to be open (no encryption)" else "Connected to an open Wi-Fi network"
                showToast(context, msg)
                NotificationHelper(context).createNotification(title, msg)
            } else {
                // Protected network
                val title = "✅ Secure Wi-Fi Connected"
                val msg = if (ssid.isNotBlank()) "Connected to $ssid (protected)" else "Connected to Wi-Fi (protected)"
                showToast(context, msg)
                NotificationHelper(context).createNotification(title, msg)
            }

        } catch (t: Throwable) {
            t.printStackTrace()
        }
    }
}
