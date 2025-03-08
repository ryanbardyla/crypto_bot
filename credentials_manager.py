import os
import json
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class CredentialsManager:
    def __init__(self, master_password, salt=None):
        # Generate or use provided salt
        self.salt = salt or os.urandom(16)
        
        # Derive key from password
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_password.encode()))
        self.cipher = Fernet(key)
        
    def encrypt_credentials(self, credentials):
        """Encrypt a dictionary of credentials"""
        credentials_json = json.dumps(credentials).encode()
        return self.cipher.encrypt(credentials_json)
    
    def decrypt_credentials(self, encrypted_data):
        """Decrypt and return credentials dictionary"""
        decrypted_data = self.cipher.decrypt(encrypted_data)
        return json.loads(decrypted_data)
    
    def save_credentials(self, filename, credentials):
        """Save encrypted credentials to file"""
        encrypted = self.encrypt_credentials(credentials)
        with open(filename, 'wb') as f:
            f.write(self.salt + b'\n' + encrypted)
    
    def load_credentials(self, filename):
        """Load encrypted credentials from file"""
        with open(filename, 'rb') as f:
            salt_line = f.readline().strip()
            encrypted = f.read()
        
        # If this is a different salt, create a new instance with the correct salt
        if salt_line != self.salt:
            self.__init__(self._master_password, salt_line)
            
        return self.decrypt_credentials(encrypted)