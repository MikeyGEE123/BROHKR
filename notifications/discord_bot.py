# notifications/discord_bot.py

import os
import asyncio
import logging
import json
import re
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from datetime import datetime
import time

import discord
from discord import app_commands
from discord.ext import commands, tasks
import aiohttp

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("discord_bot")

class BROKRDiscordBot:
    """
    Discord bot for BROKR to enable trading and notifications.
    
    This bot lets users:
    - Check token security
    - Monitor prices
    - Receive real-time alerts for buys and sells
    - Execute trades
    """
    
    def __init__(self, token: str, api_url: str = "http://localhost:8000"):
        """
        Initialize the Discord bot.
        
        Args:
            token (str): Discord bot token.
            api_url (str): URL of the BROKR API server.
        """
        self.token = token
        self.api_url = api_url
        
        # Setup intents
        intents = discord.Intents.default()
        intents.message_content = True
        
        # Create bot client
        self.bot = commands.Bot(command_prefix="!", intents=intents)
        
        # Track authorized users and authorized channels
        self.authorized_users: Set[int] = set()
        self.alert_channels: Dict[int, str] = {}  # guild_id -> channel_id
        
        # Load configuration if available
        config_path = "config/discord_bot.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.authorized_users = set(config.get('authorized_users', []))
                    self.alert_channels = config.get('alert_channels', {})
            except Exception as e:
                logger.error(f"Error loading Discord bot config: {e}")
        
        # Set up event handlers
        self.setup_event_handlers()
        
        # Set up command groups
        self.setup_commands()
    
    def setup_event_handlers(self):
        """Set up Discord event handlers."""
        
        @self.bot.event
        async def on_ready():
            guild_count = len(self.bot.guilds)
            logger.info(f"Bot is ready! Connected to {guild_count} guilds")
            
            # Start background tasks
            # self.update_price_status.start()
            
            # Sync commands
            try:
                synced = await self.bot.tree.sync()
                logger.info(f"Synced {len(synced)} command(s)")
            except Exception as e:
                logger.error(f"Failed to sync commands: {e}")
        
        @self.bot.event
        async def on_message(message):
            # Don't respond to our own messages
            if message.author == self.bot.user:
                return
            
            # Process commands
            await self.bot.process_commands(message)
            
            # Handle regular messages
            await self.handle_message(message)
    
    def setup_commands(self):
        """Set up Discord application commands."""
        
        # Help command
        @self.bot.tree.command(name="help", description="Show available commands")
        async def help_command(interaction: discord.Interaction):
            embed = discord.Embed(
                title="BROKR Bot Commands",
                description="Here are the available commands:",
                color=discord.Color.blue()
            )
            
            embed.add_field(
                name="General Commands",
                value=(
                    "`/help` - Show this help message\n"
                    "`/authorize [code]` - Authorize yourself with access code\n"
                ),
                inline=False
            )
            
            embed.add_field(
                name="Market Data",
                value=(
                    "`/price [symbol]` - Get current price of a token\n"
                    "`/security [address]` - Check token security\n"
                ),
                inline=False
            )
            
            embed.add_field(
                name="Exchange Commands",
                value=(
                    "`/exchanges` - List configured exchanges\n"
                    "`/account [exchange]` - Get account balances\n"
                ),
                inline=False
            )
            
            embed.add_field(
                name="Trading Commands (Authorized Users Only)",
                value=(
                    "`/buy [symbol] [amount]` - Place a market buy order\n"
                    "`/sell [symbol] [amount]` - Place a market sell order\n"
                ),
                inline=False
            )
            
            embed.set_footer(text="BROKR - Cryptocurrency Trading Bot")
            
            await interaction.response.send_message(embed=embed, ephemeral=True)
        
        # Authorize command
        @self.bot.tree.command(name="authorize", description="Authorize yourself to use trading features")
        @app_commands.describe(code="Your access code", private="Make response private (default: True)")
        async def authorize_command(interaction: discord.Interaction, code: str, private: bool = True):
            user_id = interaction.user.id
            
            # Check if already authorized
            if user_id in self.authorized_users:
                await interaction.response.send_message("✅ You are already authorized!", ephemeral=private)
                return
            
            # In a real implementation, you would verify the access code
            # For this example, use a simple hardcoded code "brokr123"
            if code == "brokr123":
                self.authorized_users.add(user_id)
                self.save_config()
                await interaction.response.send_message("✅ Authorization successful! You now have access to trading commands.", ephemeral=private)
            else:
                await interaction.response.send_message("❌ Invalid access code. Authorization failed.", ephemeral=private)
        
        # Price command
        @self.bot.tree.command(name="price", description="Get current price of a token")
        @app_commands.describe(symbol="Trading pair symbol (e.g., BTC/USDT)", exchange="Exchange to use (optional)")
        async def price_command(interaction: discord.Interaction, symbol: str, exchange: Optional[str] = None):
            await interaction.response.defer()
            
            try:
                url = f"{self.api_url}/market/ticker/{symbol}"
                if exchange:
                    url += f"?exchange_id={exchange}"
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            exchange_name = data.get("exchange", "Unknown")
                            price = data.get("price", 0)
                            timestamp = data.get("timestamp", 0)
                            
                            # Format as human-readable date
                            if timestamp:
                                date_str = datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d %H:%M:%S')
                            else:
                                date_str = "Unknown"
                            
                            embed = discord.Embed(
                                title=f"Price for {symbol}",
                                color=discord.Color.green()
                            )
                            
                            embed.add_field(name="Exchange", value=exchange_name, inline=True)
                            embed.add_field(name="Price", value=f"${price:,.2f}", inline=True)
                            embed.add_field(name="Time", value=date_str, inline=True)
                            
                            await interaction.followup.send(embed=embed)
                        else:
                            error_data = await response.json()
                            await interaction.followup.send(f"❌ Error getting price: {error_data.get('detail', 'Unknown error')}")
            except Exception as e:
                logger.error(f"Error getting price: {e}")
                await interaction.followup.send(f"❌ Error getting price: {str(e)}")
        
        # Security check command
        @self.bot.tree.command(name="security", description="Check token security")
        @app_commands.describe(address="Token address (e.g., 0x...)")
        async def security_command(interaction: discord.Interaction, address: str):
            await interaction.response.defer()
            
            # Validate address format
            if not re.match(r'^0x[a-fA-F0-9]{40}$', address):
                await interaction.followup.send("❌ Invalid token address format. It should be a 42-character hex string starting with 0x.")
                return
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.api_url}/security/token/check",
                        json={"token_address": address, "chain_id": 1}
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            is_safe = data.get("is_safe", False)
                            reasons = data.get("reasons", [])
                            details = data.get("details", {})
                            
                            token_info = details.get("token_info", {})
                            token_name = token_info.get("name", "Unknown Token")
                            token_symbol = token_info.get("symbol", "???")
                            
                            scores = details.get("scores", {})
                            sol_sniffer = scores.get("sol_sniffer", 0)
                            rug_check = scores.get("rug_check", 0)
                            overall = scores.get("overall", 0)
                            
                            # Create embed
                            embed = discord.Embed(
                                title=f"Security Check: {token_name} ({token_symbol})",
                                description=f"Token: {address[:6]}...{address[-4:]}\nStatus: {'✅ SAFE' if is_safe else '⚠️ POTENTIALLY UNSAFE'}",
                                color=discord.Color.green() if is_safe else discord.Color.red()
                            )
                            
                            # Add scores
                            embed.add_field(name="Overall Score", value=f"{overall:.1f}/100", inline=True)
                            embed.add_field(name="SolSniffer Score", value=f"{sol_sniffer}/100", inline=True)
                            embed.add_field(name="RugCheck Score", value=f"{rug_check}/100", inline=True)
                            
                            # Add token info
                            embed.add_field(
                                name="Token Info",
                                value=(
                                    f"Name: {token_info.get('name', 'Unknown')}\n"
                                    f"Symbol: {token_info.get('symbol', 'Unknown')}\n"
                                    f"Decimals: {token_info.get('decimals', 'Unknown')}\n"
                                    f"Total Supply: {token_info.get('total_supply', 'Unknown')}"
                                ),
                                inline=False
                            )
                            
                            # Add reasons if not safe
                            if not is_safe and reasons:
                                embed.add_field(
                                    name="Warning Reasons",
                                    value="\n".join([f"- {reason}" for reason in reasons]),
                                    inline=False
                                )
                            
                            # Add contract checks
                            checks = details.get("checks", {})
                            check_str = ""
                            
                            for check, value in checks.items():
                                emoji = "✅" if not value else "⚠️"
                                if check in ["liquidity_locked"]:
                                    emoji = "✅" if value else "⚠️"
                                
                                # Format the check name for display
                                formatted_check = check.replace("_", " ").title()
                                
                                check_str += f"{emoji} {formatted_check}\n"
                            
                            if check_str:
                                embed.add_field(name="Contract Checks", value=check_str, inline=False)
                            
                            await interaction.followup.send(embed=embed)
                        else:
                            error_data = await response.json()
                            await interaction.followup.send(f"❌ Error checking token security: {error_data.get('detail', 'Unknown error')}")
            except Exception as e:
                logger.error(f"Error checking security: {e}")
                await interaction.followup.send(f"❌ Error checking security: {str(e)}")
        
        # Exchanges command
        @self.bot.tree.command(name="exchanges", description="List configured exchanges")
        async def exchanges_command(interaction: discord.Interaction):
            await interaction.response.defer()
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.api_url}/exchanges") as response:
                        if response.status == 200:
                            exchanges = await response.json()
                            
                            if not exchanges:
                                await interaction.followup.send("ℹ️ No exchanges are configured.")
                                return
                            
                            embed = discord.Embed(
                                title="Configured Exchanges",
                                color=discord.Color.blue()
                            )
                            
                            for exchange in exchanges:
                                exchange_id = exchange.get("id", "Unknown")
                                exchange_name = exchange.get("name", "Unknown")
                                exchange_type = exchange.get("type", "Unknown")
                                is_default = exchange.get("is_default", False)
                                testnet = exchange.get("testnet", True)
                                
                                value = (
                                    f"Type: {exchange_type}\n"
                                    f"Name: {exchange_name}\n"
                                    f"Mode: {'Testnet' if testnet else 'Mainnet'}"
                                )
                                
                                embed.add_field(
                                    name=f"{exchange_id}" + (" (Default)" if is_default else ""),
                                    value=value,
                                    inline=False
                                )
                            
                            await interaction.followup.send(embed=embed)
                        else:
                            error_data = await response.json()
                            await interaction.followup.send(f"❌ Error getting exchanges: {error_data.get('detail', 'Unknown error')}")
            except Exception as e:
                logger.error(f"Error getting exchanges: {e}")
                await interaction.followup.send(f"❌ Error getting exchanges: {str(e)}")
        
        # Account command
        @self.bot.tree.command(name="account", description="Get account balances")
        @app_commands.describe(exchange="Exchange to use (optional)")
        async def account_command(interaction: discord.Interaction, exchange: Optional[str] = None):
            # Check if user is authorized
            if interaction.user.id not in self.authorized_users:
                await interaction.response.send_message(
                    "⚠️ You are not authorized to view account information.\n"
                    "Please use /authorize with your access code to gain access.",
                    ephemeral=True
                )
                return
            
            await interaction.response.defer(ephemeral=True)
            
            # Construct API URL
            url = f"{self.api_url}/trading/account"
            if exchange:
                url += f"?exchange_id={exchange}"
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            exchange_name = data.get("exchange", "Unknown")
                            balances = data.get("balances", [])
                            
                            if not balances:
                                await interaction.followup.send(f"ℹ️ No balances found for {exchange_name} exchange.", ephemeral=True)
                                return
                            
                            embed = discord.Embed(
                                title=f"Account Balances for {exchange_name}",
                                color=discord.Color.gold()
                            )
                            
                            for balance in balances:
                                asset = balance.get("asset", "Unknown")
                                free = balance.get("free", 0)
                                locked = balance.get("locked", 0)
                                total = balance.get("total", free + locked)
                                
                                value = (
                                    f"Free: {free:.8f}\n"
                                    f"Locked: {locked:.8f}\n"
                                    f"Total: {total:.8f}"
                                )
                                
                                embed.add_field(name=asset, value=value, inline=True)
                            
                            await interaction.followup.send(embed=embed, ephemeral=True)
                        else:
                            error_data = await response.json()
                            await interaction.followup.send(f"❌ Error getting account info: {error_data.get('detail', 'Unknown error')}", ephemeral=True)
            except Exception as e:
                logger.error(f"Error getting account info: {e}")
                await interaction.followup.send(f"❌ Error getting account info: {str(e)}", ephemeral=True)
        
        # Set alert channel command
        @self.bot.tree.command(name="setalerts", description="Set the channel for BROKR alerts")
        @app_commands.describe(channel="Channel to receive alerts (default: current channel)")
        @commands.has_permissions(administrator=True)
        async def set_alerts_command(interaction: discord.Interaction, channel: Optional[discord.TextChannel] = None):
            # Use the current channel if none specified
            alert_channel = channel or interaction.channel
            
            # Save the alert channel for this guild
            self.alert_channels[str(interaction.guild_id)] = str(alert_channel.id)
            self.save_config()
            
            await interaction.response.send_message(f"✅ Alert channel set to {alert_channel.mention}")
    
    async def handle_message(self, message):
        """Handle regular messages."""
        # Ignore bot messages
        if message.author.bot:
            return
        
        content = message.content.lower()
        
        # Check for token addresses
        token_address_match = re.search(r'0x[a-fA-F0-9]{40}', content)
        if token_address_match:
            token_address = token_address_match.group(0)
            
            # Create button for security check
            view = discord.ui.View()
            button = discord.ui.Button(label="Check Token Security", style=discord.ButtonStyle.primary)
            
            async def button_callback(interaction):
                # Defer the response since the security check might take time
                await interaction.response.defer()
                
                # Call the security command function
                command = self.bot.tree.get_command("security")
                if command:
                    ctx = await self.bot.get_context(message)
                    await command._callback(interaction, address=token_address)
            
            button.callback = button_callback
            view.add_item(button)
            
            await message.reply("I noticed you shared a token address. Would you like to check its security?", view=view)
            return
        
        # Check for trading pairs
        trading_pair_match = re.search(r'([A-Za-z0-9]+)/([A-Za-z0-9]+)', content)
        if trading_pair_match:
            symbol = trading_pair_match.group(0).upper()
            
            # Create button for price check
            view = discord.ui.View()
            button = discord.ui.Button(label="Check Price", style=discord.ButtonStyle.primary)
            
            async def button_callback(interaction):
                # Defer the response since the API call might take time
                await interaction.response.defer()
                
                # Call the price command function
                command = self.bot.tree.get_command("price")
                if command:
                    ctx = await self.bot.get_context(message)
                    await command._callback(interaction, symbol=symbol)
            
            button.callback = button_callback
            view.add_item(button)
            
            await message.reply(f"I noticed you mentioned {symbol}. Would you like to check its price?", view=view)
            return
    
    @tasks.loop(minutes=5)
    async def update_price_status(self):
        """Update bot status with BTC price."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/market/ticker/BTC/USDT") as response:
                    if response.status == 200:
                        data = await response.json()
                        price = data.get("price", 0)
                        await self.bot.change_presence(
                            activity=discord.Activity(
                                type=discord.ActivityType.watching,
                                name=f"BTC: ${price:,.2f}"
                            )
                        )
        except Exception as e:
            logger.error(f"Error updating price status: {e}")
    
    def save_config(self):
        """Save the bot configuration to a file."""
        config_path = "config/discord_bot.json"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                json.dump({
                    'authorized_users': list(self.authorized_users),
                    'alert_channels': self.alert_channels
                }, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving Discord bot config: {e}")
    
    async def send_alert(self, guild_id: int, message: str, embed=None):
        """Send an alert to the designated channel in a guild."""
        if not self.bot:
            logger.error("Bot not initialized. Cannot send alert.")
            return False
        
        guild_id_str = str(guild_id)
        if guild_id_str not in self.alert_channels:
            logger.warning(f"No alert channel configured for guild {guild_id}")
            return False
        
        try:
            channel_id = int(self.alert_channels[guild_id_str])
            channel = self.bot.get_channel(channel_id)
            if not channel:
                logger.warning(f"Alert channel {channel_id} not found")
                return False
            
            if embed:
                await channel.send(message, embed=embed)
            else:
                await channel.send(message)
            return True
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            return False
    
    async def broadcast_alert(self, message: str, embed=None):
        """Broadcast an alert to all configured alert channels."""
        if not self.bot:
            logger.error("Bot not initialized. Cannot broadcast alert.")
            return False
        
        success_count = 0
        for guild_id in self.alert_channels:
            success = await self.send_alert(int(guild_id), message, embed)
            if success:
                success_count += 1
        
        return success_count
    
    def run(self):
        """Run the Discord bot."""
        self.bot.run(self.token)
    
    async def close(self):
        """Close the Discord bot."""
        if self.bot:
            await self.bot.close()

# Helper function to create and run the bot
def run_discord_bot(token: str, api_url: str = "http://localhost:8000"):
    """Create and run the Discord bot."""
    bot = BROKRDiscordBot(token, api_url)
    bot.run()

if __name__ == "__main__":
    # Example: Replace with your actual bot token from Discord Developer Portal
    token = os.environ.get("DISCORD_BOT_TOKEN", "your_bot_token_here")
    api_url = os.environ.get("BROKR_API_URL", "http://localhost:8000")
    
    run_discord_bot(token, api_url)
