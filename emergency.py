# emergency.py
import threading
import smtplib
from email.mime.text import MIMEText
from utils import get_location_ip
from config import GMAIL_USER, GMAIL_PASSWORD, EMERGENCY_CONTACTS

def send_email_alert():
    try:
        loc, city, region, country = get_location_ip()
        subject = "🚨 VeeAssist Emergency Alert"
        if loc:
            body = f"EMERGENCY DETECTED!\n\nUser may need immediate help.\n\nApproximate Location: Nagapatla \nCoordinates: https://maps.app.goo.gl/XzTcAfiQxcvSVDST6\n\nThis alert was triggered by VeeAssist navigation system."
        else:
            body = "EMERGENCY DETECTED!\n\nUser may need immediate help.\n\nLocation could not be determined.\n\nThis alert was triggered by VeeAssist navigation system."

        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = GMAIL_USER

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(GMAIL_USER, GMAIL_PASSWORD)

        for contact in EMERGENCY_CONTACTS:
            msg['To'] = contact
            server.sendmail(GMAIL_USER, contact, msg.as_string())
            print(f"✅ Emergency email sent to {contact}")

        server.quit()
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False

def trigger_emergency_alert(set_emergency_flag, set_narration):
    if not set_emergency_flag():
        set_emergency_flag(True)
        print("🚨 EMERGENCY ALERT TRIGGERED!")
        set_narration("Emergency detected! Camera blackout. Sending alert to emergency contacts.")
        threading.Thread(target=send_email_alert, daemon=True).start()
