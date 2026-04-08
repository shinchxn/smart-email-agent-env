from huggingface_hub import HfApi
import sys

def check_token():
    api = HfApi()
    try:
        user_info = api.whoami()
        print(f"Authenticated as: {user_info['name']}")
        
        # Check if the token has write access
        # The 'auth' key in whoami usually contains token info, but simpler to check the 'canPay' or specific permissions
        # Actually, in newer versions, whoami returns detailed info
        token_type = "read"
        if "write" in user_info.get("auth", {}).get("accessToken", {}).get("role", "").lower():
            token_type = "write"
        elif user_info.get("auth", {}).get("type", "") == "write":
            token_type = "write"
            
        print(f"Token Type: {token_type}")
        
        if token_type == "read":
            print("\n!!! ERROR: Your token only has READ access. !!!")
            print("Please generate a WRITE token at https://huggingface.co/settings/tokens")
        else:
            print("\n✓ SUCCESS: Your token has WRITE access.")
            
    except Exception as e:
        print(f"Error checking token: {e}")
        print("\nYou are likely not logged in. Run: python -m huggingface_hub.cli login")

if __name__ == "__main__":
    check_token()
