# security/token_security.py

import json
import os
import logging
import asyncio
import aiohttp
import re
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from datetime import datetime
import hashlib

# Set up logging
logger = logging.getLogger(__name__)

class TokenSecurity:
    """
    Token security class for filtering and blacklisting tokens.
    
    This class provides methods to check tokens against various security criteria:
    - Token blacklist
    - Developer blacklist
    - Contract code analysis
    - Liquidity lock verification
    - Token supply distribution analysis
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the token security system.
        
        Args:
            config_path (str, optional): Path to a JSON configuration file.
                If provided, blacklists will be loaded from this config.
        """
        # Default blacklists
        self.token_blacklist: Set[str] = set()  # Set of token addresses
        self.dev_blacklist: Set[str] = set()    # Set of developer addresses
        
        # Configuration for security checks
        self.min_sol_sniffer_score = 80
        self.check_liquidity_lock = True
        self.check_bundled_supply = True
        
        # Cached security results
        self.security_results_cache: Dict[str, Dict[str, Any]] = {}
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

    def load_config(self, config_path: str) -> bool:
        """
        Load security configuration from a JSON file.
        
        Args:
            config_path (str): Path to a JSON configuration file.
            
        Returns:
            bool: True if loading was successful, False otherwise.
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Load blacklists
            self.token_blacklist = set(config.get('token_blacklist', []))
            self.dev_blacklist = set(config.get('dev_blacklist', []))
            
            # Load security settings
            security_settings = config.get('security_settings', {})
            self.min_sol_sniffer_score = security_settings.get('min_sol_sniffer_score', self.min_sol_sniffer_score)
            self.check_liquidity_lock = security_settings.get('check_liquidity_lock', self.check_liquidity_lock)
            self.check_bundled_supply = security_settings.get('check_bundled_supply', self.check_bundled_supply)
            
            return True
        except Exception as e:
            logger.error(f"Error loading security config: {e}")
            return False
    
    def save_config(self, config_path: str) -> bool:
        """
        Save current security configuration to a JSON file.
        
        Args:
            config_path (str): Path to save the configuration.
            
        Returns:
            bool: True if saving was successful, False otherwise.
        """
        try:
            config = {
                'token_blacklist': list(self.token_blacklist),
                'dev_blacklist': list(self.dev_blacklist),
                'security_settings': {
                    'min_sol_sniffer_score': self.min_sol_sniffer_score,
                    'check_liquidity_lock': self.check_liquidity_lock,
                    'check_bundled_supply': self.check_bundled_supply
                }
            }
            
            # Write to file
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            return True
        except Exception as e:
            logger.error(f"Error saving security config: {e}")
            return False
    
    def add_to_token_blacklist(self, token_address: str) -> bool:
        """
        Add a token to the blacklist.
        
        Args:
            token_address (str): The token address to blacklist.
            
        Returns:
            bool: True if the token was added, False if it was already blacklisted.
        """
        token_address = token_address.lower()
        if token_address in self.token_blacklist:
            return False
        
        self.token_blacklist.add(token_address)
        return True
    
    def remove_from_token_blacklist(self, token_address: str) -> bool:
        """
        Remove a token from the blacklist.
        
        Args:
            token_address (str): The token address to remove.
            
        Returns:
            bool: True if the token was removed, False if it wasn't blacklisted.
        """
        token_address = token_address.lower()
        if token_address not in self.token_blacklist:
            return False
        
        self.token_blacklist.remove(token_address)
        return True
    
    def add_to_dev_blacklist(self, developer_address: str) -> bool:
        """
        Add a developer to the blacklist.
        
        Args:
            developer_address (str): The developer address to blacklist.
            
        Returns:
            bool: True if the developer was added, False if already blacklisted.
        """
        developer_address = developer_address.lower()
        if developer_address in self.dev_blacklist:
            return False
        
        self.dev_blacklist.add(developer_address)
        return True
    
    def remove_from_dev_blacklist(self, developer_address: str) -> bool:
        """
        Remove a developer from the blacklist.
        
        Args:
            developer_address (str): The developer address to remove.
            
        Returns:
            bool: True if the developer was removed, False if not blacklisted.
        """
        developer_address = developer_address.lower()
        if developer_address not in self.dev_blacklist:
            return False
        
        self.dev_blacklist.remove(developer_address)
        return True
    
    def get_token_blacklist(self) -> List[str]:
        """
        Get the list of blacklisted tokens.
        
        Returns:
            List[str]: List of blacklisted token addresses.
        """
        return list(self.token_blacklist)
    
    def get_dev_blacklist(self) -> List[str]:
        """
        Get the list of blacklisted developers.
        
        Returns:
            List[str]: List of blacklisted developer addresses.
        """
        return list(self.dev_blacklist)
    
    def is_token_blacklisted(self, token_address: str) -> bool:
        """
        Check if a token is blacklisted.
        
        Args:
            token_address (str): The token address to check.
            
        Returns:
            bool: True if the token is blacklisted, False otherwise.
        """
        return token_address.lower() in self.token_blacklist
    
    def is_developer_blacklisted(self, developer_address: str) -> bool:
        """
        Check if a developer is blacklisted.
        
        Args:
            developer_address (str): The developer address to check.
            
        Returns:
            bool: True if the developer is blacklisted, False otherwise.
        """
        return developer_address.lower() in self.dev_blacklist
    
    def clear_token_blacklist(self) -> None:
        """Clear the token blacklist."""
        self.token_blacklist.clear()
    
    def clear_dev_blacklist(self) -> None:
        """Clear the developer blacklist."""
        self.dev_blacklist.clear()
    
    def set_min_sol_sniffer_score(self, score: int) -> None:
        """
        Set the minimum SolSniffer score for tokens.
        
        Args:
            score (int): Minimum score (0-100).
        """
        self.min_sol_sniffer_score = max(0, min(100, score))
    
    def set_check_liquidity_lock(self, enabled: bool) -> None:
        """
        Set whether to check liquidity locks.
        
        Args:
            enabled (bool): Whether to check liquidity locks.
        """
        self.check_liquidity_lock = enabled
    
    def set_check_bundled_supply(self, enabled: bool) -> None:
        """
        Set whether to check for bundled supplies.
        
        Args:
            enabled (bool): Whether to check for bundled supplies.
        """
        self.check_bundled_supply = enabled
    
    async def check_token_security(self, token_address: str, chain_id: int = 1, 
                                force_refresh: bool = False) -> Dict[str, Any]:
        """
        Comprehensive token security check.
        
        Args:
            token_address (str): The token address to check.
            chain_id (int): The blockchain ID (1 for Ethereum mainnet).
            force_refresh (bool): Whether to force a refresh of cached results.
            
        Returns:
            Dict[str, Any]: Security check results.
        """
        token_address = token_address.lower()
        cache_key = f"{token_address}_{chain_id}"
        
        # Check cache unless force refresh is requested
        if not force_refresh and cache_key in self.security_results_cache:
            return self.security_results_cache[cache_key]
        
        # Initialize result structure
        result = {
            "token_address": token_address,
            "chain_id": chain_id,
            "timestamp": datetime.now().isoformat(),
            "is_blacklisted": self.is_token_blacklisted(token_address),
            "developer": {
                "address": None,
                "is_blacklisted": False
            },
            "scores": {
                "overall": 0,
                "sol_sniffer": 0,
                "rug_check": 0
            },
            "checks": {
                "liquidity_locked": False,
                "bundled_supply": False,
                "honeypot": False,
                "hidden_owner": False,
                "proxy_contract": False,
                "can_take_back_ownership": False,
                "is_mintable": False,
                "has_trading_cooldown": False
            },
            "warnings": [],
            "token_info": {
                "name": None,
                "symbol": None,
                "decimals": None,
                "total_supply": None,
                "creator": None,
                "creation_time": None
            }
        }
        
        try:
            # Run all security checks concurrently
            developer_info, sol_sniffer_result, rug_check_result, token_info = await asyncio.gather(
                self._get_token_developer(token_address, chain_id),
                self._check_sol_sniffer(token_address, chain_id),
                self._check_rug_check(token_address, chain_id),
                self._get_token_info(token_address, chain_id)
            )
            
            # Process developer information
            if developer_info:
                result["developer"]["address"] = developer_info.get("address")
                result["developer"]["is_blacklisted"] = self.is_developer_blacklisted(developer_info.get("address", ""))
            
            # Process SolSniffer score
            if sol_sniffer_result:
                result["scores"]["sol_sniffer"] = sol_sniffer_result.get("score", 0)
                
                # Add SolSniffer specific checks
                for check, value in sol_sniffer_result.get("checks", {}).items():
                    if check in result["checks"]:
                        result["checks"][check] = value
                
                # Add warnings from SolSniffer
                result["warnings"].extend(sol_sniffer_result.get("warnings", []))
            
            # Process RugCheck results
            if rug_check_result:
                result["scores"]["rug_check"] = rug_check_result.get("score", 0)
                
                # Add RugCheck specific checks
                for check, value in rug_check_result.get("checks", {}).items():
                    if check in result["checks"]:
                        result["checks"][check] = value
                
                # Add warnings from RugCheck
                result["warnings"].extend(rug_check_result.get("warnings", []))
            
            # Process token info
            if token_info:
                result["token_info"] = token_info
            
            # Calculate overall score based on both services
            if sol_sniffer_result and rug_check_result:
                result["scores"]["overall"] = (
                    result["scores"]["sol_sniffer"] * 0.6 + 
                    result["scores"]["rug_check"] * 0.4
                )
            elif sol_sniffer_result:
                result["scores"]["overall"] = result["scores"]["sol_sniffer"]
            elif rug_check_result:
                result["scores"]["overall"] = result["scores"]["rug_check"]
            
            # Add custom warnings based on configuration
            if self.min_sol_sniffer_score > 0 and result["scores"]["sol_sniffer"] < self.min_sol_sniffer_score:
                result["warnings"].append(
                    f"SolSniffer score ({result['scores']['sol_sniffer']}) is below minimum threshold ({self.min_sol_sniffer_score})"
                )
            
            if self.check_liquidity_lock and not result["checks"]["liquidity_locked"]:
                result["warnings"].append("Liquidity is not locked")
            
            if self.check_bundled_supply and result["checks"]["bundled_supply"]:
                result["warnings"].append("Token has bundled supply (potential rug risk)")
            
            # Update cache
            self.security_results_cache[cache_key] = result
            
            return result
        except Exception as e:
            logger.error(f"Error checking token security for {token_address}: {e}")
            result["error"] = str(e)
            return result
    
    async def _get_token_developer(self, token_address: str, chain_id: int) -> Dict[str, Any]:
        """
        Get information about the token developer.
        
        Args:
            token_address (str): The token address to check.
            chain_id (int): The blockchain ID.
            
        Returns:
            Dict[str, Any]: Developer information.
        """
        # This would typically involve blockchain API calls to get the contract creator
        # For this example, we'll return mock data
        try:
            # Deterministic mock data based on token address
            developer_hash = hashlib.md5(token_address.encode()).hexdigest()
            mock_developer = f"0x{developer_hash[:40]}"
            
            return {
                "address": mock_developer,
                "creation_block": 12345678,
                "creation_time": "2023-01-01T00:00:00Z",
                "tx_hash": f"0x{developer_hash}"
            }
        except Exception as e:
            logger.error(f"Error getting token developer: {e}")
            return {}
    
    async def _check_sol_sniffer(self, token_address: str, chain_id: int) -> Dict[str, Any]:
        """
        Check the token with SolSniffer (mock implementation).
        
        Args:
            token_address (str): The token address to check.
            chain_id (int): The blockchain ID.
            
        Returns:
            Dict[str, Any]: SolSniffer check results.
        """
        # This would typically involve API calls to SolSniffer
        # For this example, we'll return mock data
        try:
            # Deterministic score based on token address
            address_num = int(token_address[-8:], 16)
            mock_score = (address_num % 100)
            
            return {
                "score": mock_score,
                "checks": {
                    "liquidity_locked": mock_score > 70,
                    "bundled_supply": mock_score < 30,
                    "honeypot": mock_score < 20,
                    "hidden_owner": mock_score < 50,
                    "proxy_contract": mock_score < 60,
                    "can_take_back_ownership": mock_score < 40,
                    "is_mintable": mock_score < 70,
                    "has_trading_cooldown": mock_score < 80,
                },
                "warnings": [
                    "This is a mock SolSniffer check for demonstration"
                ] + (
                    ["Low score: potential scam risk"] if mock_score < 50 else []
                )
            }
        except Exception as e:
            logger.error(f"Error checking SolSniffer: {e}")
            return {}
    
    async def _check_rug_check(self, token_address: str, chain_id: int) -> Dict[str, Any]:
        """
        Check the token with RugCheck (mock implementation).
        
        Args:
            token_address (str): The token address to check.
            chain_id (int): The blockchain ID.
            
        Returns:
            Dict[str, Any]: RugCheck check results.
        """
        # This would typically involve API calls to RugCheck
        # For this example, we'll return mock data
        try:
            # Slightly different deterministic score based on token address
            address_num = int(token_address[-10:], 16)
            mock_score = (address_num % 90) + 10
            
            return {
                "score": mock_score,
                "classification": "good" if mock_score >= 80 else "suspicious" if mock_score >= 50 else "dangerous",
                "checks": {
                    "liquidity_locked": mock_score > 60,
                    "hidden_owner": mock_score < 60,
                    "proxy_contract": mock_score < 70,
                    "can_take_back_ownership": mock_score < 50,
                },
                "warnings": [
                    "This is a mock RugCheck check for demonstration"
                ] + (
                    ["Potentially unsafe contract"] if mock_score < 60 else []
                )
            }
        except Exception as e:
            logger.error(f"Error checking RugCheck: {e}")
            return {}
    
    async def _get_token_info(self, token_address: str, chain_id: int) -> Dict[str, Any]:
        """
        Get basic token information.
        
        Args:
            token_address (str): The token address to check.
            chain_id (int): The blockchain ID.
            
        Returns:
            Dict[str, Any]: Token information.
        """
        # This would typically involve blockchain API calls to get token information
        # For this example, we'll return mock data
        try:
            # Mock token info based on address hash
            token_hash = hashlib.md5(token_address.encode()).hexdigest()
            first_chars = token_hash[:6].upper()
            
            return {
                "name": f"Mock Token {first_chars}",
                "symbol": f"MT{first_chars[:3]}",
                "decimals": 18,
                "total_supply": 1000000000,
                "creator": f"0x{token_hash[:40]}",
                "creation_time": "2023-01-01T00:00:00Z"
            }
        except Exception as e:
            logger.error(f"Error getting token info: {e}")
            return {}
    
    async def scan_tokens(self, token_addresses: List[str], chain_id: int = 1) -> Dict[str, Dict[str, Any]]:
        """
        Scan multiple tokens for security issues.
        
        Args:
            token_addresses (List[str]): List of token addresses to scan.
            chain_id (int): The blockchain ID.
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping token addresses to security check results.
        """
        results = {}
        
        # Check tokens concurrently
        tasks = [self.check_token_security(addr, chain_id) for addr in token_addresses]
        token_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for token_address, result in zip(token_addresses, token_results):
            if isinstance(result, Exception):
                logger.error(f"Error scanning token {token_address}: {result}")
                results[token_address] = {"error": str(result)}
            else:
                results[token_address] = result
        
        return results
    
    def is_token_safe(self, security_result: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Determine if a token is safe based on security check results.
        
        Args:
            security_result (Dict[str, Any]): Security check result from check_token_security.
            
        Returns:
            Tuple[bool, List[str]]: (is_safe, reasons)
        """
        is_safe = True
        reasons = []
        
        # Check if token or developer is blacklisted
        if security_result["is_blacklisted"]:
            is_safe = False
            reasons.append("Token is blacklisted")
        
        if security_result["developer"]["is_blacklisted"]:
            is_safe = False
            reasons.append("Developer is blacklisted")
        
        # Check SolSniffer score
        if security_result["scores"]["sol_sniffer"] < self.min_sol_sniffer_score:
            is_safe = False
            reasons.append(f"SolSniffer score is below threshold: {security_result['scores']['sol_sniffer']} < {self.min_sol_sniffer_score}")
        
        # Check if liquidity is locked (if configured)
        if self.check_liquidity_lock and not security_result["checks"]["liquidity_locked"]:
            is_safe = False
            reasons.append("Liquidity is not locked")
        
        # Check for bundled supply (if configured)
        if self.check_bundled_supply and security_result["checks"]["bundled_supply"]:
            is_safe = False
            reasons.append("Token has bundled supply")
        
        # Check for dangerous contract features
        if security_result["checks"]["honeypot"]:
            is_safe = False
            reasons.append("Token is a potential honeypot")
        
        if security_result["checks"]["hidden_owner"]:
            is_safe = False
            reasons.append("Token has hidden owner function")
        
        if security_result["checks"]["can_take_back_ownership"]:
            is_safe = False
            reasons.append("Developer can take back ownership")
        
        return is_safe, reasons

# Example usage
async def example_usage():
    # Create token security
    security = TokenSecurity()
    
    # Add some addresses to blacklists
    security.add_to_token_blacklist("0x1234567890123456789012345678901234567890")
    security.add_to_dev_blacklist("0xabcdefabcdefabcdefabcdefabcdefabcdefabcd")
    
    # Configure security settings
    security.set_min_sol_sniffer_score(80)
    security.set_check_liquidity_lock(True)
    security.set_check_bundled_supply(True)
    
    # Check a token
    result = await security.check_token_security("0x0000000000000000000000000000000000000000")
    is_safe, reasons = security.is_token_safe(result)
    
    # Print results
    print(f"Token safe: {is_safe}")
    if not is_safe:
        print("Reasons:")
        for reason in reasons:
            print(f"- {reason}")
    
    # Save configuration
    security.save_config("token_security_config.json")

if __name__ == "__main__":
    asyncio.run(example_usage())
