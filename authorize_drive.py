"""
One-time Google Drive authorization (OAuth) so the pipeline can upload finished
videos to your Drive AS YOU (service accounts have no storage quota and can't
write to a normal "My Drive").

Setup (once):
  1. Google Cloud Console → same project → APIs & Services → Credentials →
     Create Credentials → OAuth client ID → Application type: "Desktop app".
     Download the JSON and save it here as  client_secret.json
  2. OAuth consent screen: add your Google account as a Test user (and the
     .../auth/drive scope). Publishing the app avoids the 7-day token expiry.
  3. Run:   ./.venv/bin/python authorize_drive.py
     A browser opens → sign in with the account that has access to the batch
     folders → allow. A token.json is written; the server uses it automatically.
"""
import os

from google_auth_oauthlib.flow import InstalledAppFlow

ROOT = os.path.dirname(os.path.abspath(__file__))
SCOPES = ["https://www.googleapis.com/auth/drive"]


def main() -> None:
    secret = os.path.join(ROOT, "client_secret.json")
    if not os.path.exists(secret):
        raise SystemExit("Missing client_secret.json (OAuth Desktop credentials) "
                         "— see the setup steps at the top of this file.")
    flow = InstalledAppFlow.from_client_secrets_file(secret, SCOPES)
    creds = flow.run_local_server(port=0)
    with open(os.path.join(ROOT, "token.json"), "w") as f:
        f.write(creds.to_json())
    print("✅ Authorized. token.json saved — the pipeline can now upload to Drive.")


if __name__ == "__main__":
    main()
