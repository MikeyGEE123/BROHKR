# notifications/telegram_bot.py

import os
import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from datetime import datetime
import re

from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode

import aiohttp

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("telegram_bot")

class BROKRTelegramBot:
    """
    Telegram bot for BROKR to enable trading and notifications via chat.
    
    This bot lets users:
    - Check token security
    - Monitor prices
    - Receive real-time alerts for buys and sells
    - Execute trades
    """
    
    def __init__(self, token: str, api_url: str = "http://localhost:8000"):
        """
        Initialize the Telegram bot.
        
        Args:
            token (str): Telegram bot token from BotFather.
            api_url (str): URL of the BROKR API server.
        """
        self.token = token
        self.api_url = api_url
        self.application = None
        self.bot = None
        
        # Track authenticated users
        self.authorized_users: Set[int] = set()
        
        # Load authorized users from config if available
        config_path = "config/telegram_bot.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.authorized_users = set(config.get('authorized_users', []))
            except Exception as e:
                logger.error(f"Error loading Telegram bot config: {e}")
    
    async def initialize(self):
        """Initialize the bot and set up handlers."""
        self.bot = Bot(self.token)
        self.application = Application.builder().token(self.token).build()
        
        # Add command handlers
        self.application.add_handler(CommandHandler("start", self.cmd_start))
        self.application.add_handler(CommandHandler("help", self.cmd_help))
        self.application.add_handler(CommandHandler("authorize", self.cmd_authorize))
        self.application.add_handler(CommandHandler("price", self.cmd_price))
        self.application.add_handler(CommandHandler("securitycheck", self.cmd_security_check))
        self.application.add_handler(CommandHandler("exchanges", self.cmd_exchanges))
        self.application.add_handler(CommandHandler("account", self.cmd_account))
        
        # Add callback query handler for inline buttons
        self.application.add_handler(CallbackQueryHandler(self.button_callback))
        
        # Add message handler for any text messages
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Add error handler
        self.application.add_error_handler(self.error_handler)
        
        return self
    
    def start(self):
        """Start the bot (blocking)."""
        if self.application:
            self.application.run_polling()
    
    async def start_polling(self):
        """Start the bot (non-blocking)."""
        if self.application:
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
    
    async def stop(self):
        """Stop the bot."""
        if self.application:
            await self.application.stop()
            await self.application.shutdown()
    
    def save_config(self):
        """Save the bot configuration to a file."""
        config_path = "config/telegram_bot.json"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                json.dump({
                    'authorized_users': list(self.authorized_users)
                }, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving Telegram bot config: {e}")
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /start command."""
        user_id = update.effective_user.id
        username = update.effective_user.username or update.effective_user.first_name
        
        await update.message.reply_html(
            f"👋 Hello, {username}!\n\n"
            f"I'm the <b>BROKR</b> Telegram bot. I can help you monitor crypto prices, "
            f"check token security, and execute trades.\n\n"
            f"Type /help to see available commands."
        )
        
        # Check if user is authorized
        if user_id not in self.authorized_users:
            await update.message.reply_text(
                "⚠️ You are not authorized to use trading features.\n"
                "Please use /authorize with your access code to gain access."
            )
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /help command."""
        await update.message.reply_text(
            "🤖 *BROKR Bot Commands*\n\n"
            "*General Commands:*\n"
            "/start - Start the bot\n"
            "/help - Show this help message\n"
            "/authorize [code] - Authorize yourself with access code\n\n"
            
            "*Market Data:*\n"
            "/price [symbol] - Get current price of a token (e.g., /price BTC/USDT)\n"
            "/securitycheck [address] - Check token security (e.g., /securitycheck 0x...)\n\n"
            
            "*Exchange Commands:*\n"
            "/exchanges - List configured exchanges\n"
            "/account [exchange] - Get account balances (e.g., /account binance)\n\n"
            
            "*Trading Commands (Authorized Users Only):*\n"
            "/buy [symbol] [amount] - Place a market buy order\n"
            "/sell [symbol] [amount] - Place a market sell order\n",
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def cmd_authorize(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /authorize command."""
        user_id = update.effective_user.id
        
        # Check if already authorized
        if user_id in self.authorized_users:
            await update.message.reply_text("✅ You are already authorized!")
            return
        
        # Check if access code is provided
        if not context.args or len(context.args) != 1:
            await update.message.reply_text(
                "❌ Please provide your access code:\n"
                "/authorize [access_code]"
            )
            return
        
        access_code = context.args[0]
        
        # In a real implementation, you would verify the access code
        # For this example, use a simple hardcoded code "brokr123"
        if access_code == "brokr123":
            self.authorized_users.add(user_id)
            self.save_config()
            await update.message.reply_text(
                "✅ Authorization successful! You now have access to trading commands."
            )
        else:
            await update.message.reply_text("❌ Invalid access code. Authorization failed.")
    
    async def cmd_price(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /price command."""
        # Check if symbol is provided
        if not context.args or len(context.args) != 1:
            await update.message.reply_text(
                "❌ Please provide a trading pair symbol:\n"
                "/price [symbol] (e.g., /price BTC/USDT)"
            )
            return
        
        symbol = context.args[0].upper()
        
        # Call the BROKR API
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/market/ticker/{symbol}") as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        exchange = data.get("exchange", "Unknown")
                        price = data.get("price", 0)
                        timestamp = data.get("timestamp", 0)
                        
                        # Format as human-readable date
                        if timestamp:
                            date_str = datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            date_str = "Unknown"
                        
                        await update.message.reply_text(
                            f"💰 *Price for {symbol}*\n\n"
                            f"*Exchange:* {exchange}\n"
                            f"*Price:* ${price:,.2f}\n"
                            f"*Time:* {date_str}\n",
                            parse_mode=ParseMode.MARKDOWN
                        )
                    else:
                        error_data = await response.json()
                        await update.message.reply_text(
                            f"❌ Error getting price: {error_data.get('detail', 'Unknown error')}"
                        )
        except Exception as e:
            logger.error(f"Error getting price: {e}")
            await update.message.reply_text(f"❌ Error getting price: {str(e)}")
    
    async def cmd_security_check(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /securitycheck command."""
        # Check if token address is provided
        if not context.args or len(context.args) != 1:
            await update.message.reply_text(
                "❌ Please provide a token address:\n"
                "/securitycheck [address] (e.g., /securitycheck 0x...)"
            )
            return
        
        token_address = context.args[0].lower()
        
        # Validate address format
        if not re.match(r'^0x[a-fA-F0-9]{40}$', token_address):
            await update.message.reply_text("❌ Invalid token address format. It should be a 42-character hex string starting with 0x.")
            return
        
        await update.message.reply_text("🔍 Checking token security... This may take a moment.")
        
        # Call the BROKR API
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/security/token/check",
                    json={"token_address": token_address, "chain_id": 1}
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
                        
                        # Format security check result
                        safety_emoji = "✅" if is_safe else "⚠️"
                        response_text = (
                            f"{safety_emoji} *Security Check for {token_name} ({token_symbol})*\n\n"
                            f"*Token:* {token_address[:6]}...{token_address[-4:]}\n"
                            f"*Status:* {'SAFE' if is_safe else 'POTENTIALLY UNSAFE'}\n\n"
                            f"*Scores:*\n"
                            f"- Overall: {overall:.1f}/100\n"
                            f"- SolSniffer: {sol_sniffer}/100\n"
                            f"- RugCheck: {rug_check}/100\n\n"
                        )
                        
                        # Add reasons if the token is not safe
                        if not is_safe and reasons:
                            response_text += "*Warning Reasons:*\n"
                            for reason in reasons:
                                response_text += f"- {reason}\n"
                        
                        # Add buttons for more details
                        keyboard = [
                            [InlineKeyboardButton("View Token Info", callback_data=f"token_info:{token_address}")]
                        ]
                        
                        await update.message.reply_text(
                            response_text,
                            parse_mode=ParseMode.MARKDOWN,
                            reply_markup=InlineKeyboardMarkup(keyboard)
                        )
                    else:
                        error_data = await response.json()
                        await update.message.reply_text(
                            f"❌ Error checking token security: {error_data.get('detail', 'Unknown error')}"
                        )
        except Exception as e:
            logger.error(f"Error checking token security: {e}")
            await update.message.reply_text(f"❌ Error checking token security: {str(e)}")
    
    async def cmd_exchanges(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /exchanges command."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/exchanges") as response:
                    if response.status == 200:
                        exchanges = await response.json()
                        
                        if not exchanges:
                            await update.message.reply_text("ℹ️ No exchanges are configured.")
                            return
                        
                        response_text = "🔄 *Configured Exchanges*\n\n"
                        
                        for exchange in exchanges:
                            exchange_id = exchange.get("id", "Unknown")
                            exchange_name = exchange.get("name", "Unknown")
                            exchange_type = exchange.get("type", "Unknown")
                            is_default = exchange.get("is_default", False)
                            testnet = exchange.get("testnet", True)
                            
                            response_text += (
                                f"*{exchange_id}*" + (" (Default)" if is_default else "") + "\n"
                                f"- Type: {exchange_type}\n"
                                f"- Name: {exchange_name}\n"
                                f"- Mode: {'Testnet' if testnet else 'Mainnet'}\n\n"
                            )
                        
                        await update.message.reply_text(
                            response_text,
                            parse_mode=ParseMode.MARKDOWN
                        )
                    else:
                        error_data = await response.json()
                        await update.message.reply_text(
                            f"❌ Error getting exchanges: {error_data.get('detail', 'Unknown error')}"
                        )
        except Exception as e:
            logger.error(f"Error getting exchanges: {e}")
            await update.message.reply_text(f"❌ Error getting exchanges: {str(e)}")
    
    async def cmd_account(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /account command."""
        # Check if user is authorized
        user_id = update.effective_user.id
        if user_id not in self.authorized_users:
            await update.message.reply_text(
                "⚠️ You are not authorized to view account information.\n"
                "Please use /authorize with your access code to gain access."
            )
            return
        
        exchange_id = None
        if context.args and len(context.args) == 1:
            exchange_id = context.args[0]
        
        # Construct API URL
        url = f"{self.api_url}/trading/account"
        if exchange_id:
            url += f"?exchange_id={exchange_id}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        exchange = data.get("exchange", "Unknown")
                        balances = data.get("balances", [])
                        
                        if not balances:
                            await update.message.reply_text(f"ℹ️ No balances found for {exchange} exchange.")
                            return
                        
                        response_text = f"💼 *Account Balances for {exchange}*\n\n"
                        
                        for balance in balances:
                            asset = balance.get("asset", "Unknown")
                            free = balance.get("free", 0)
                            locked = balance.get("locked", 0)
                            total = balance.get("total", free + locked)
                            
                            response_text += (
                                f"*{asset}*\n"
                                f"- Free: {free:.8f}\n"
                                f"- Locked: {locked:.8f}\n"
                                f"- Total: {total:.8f}\n\n"
                            )
                        
                        await update.message.reply_text(
                            response_text,
                            parse_mode=ParseMode.MARKDOWN
                        )
                    else:
                        error_data = await response.json()
                        await update.message.reply_text(
                            f"❌ Error getting account info: {error_data.get('detail', 'Unknown error')}"
                        )
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            await update.message.reply_text(f"❌ Error getting account info: {str(e)}")
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks."""
        query = update.callback_query
        
        # Always answer the callback query to remove the loading state
        await query.answer()
        
        # Parse the callback data
        data = query.data
        
        if data.startswith("token_info:"):
            # Extract token address
            token_address = data.split(":", 1)[1]
            
            # Call the BROKR API to get token info
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.api_url}/security/token/check",
                        json={"token_address": token_address, "chain_id": 1}
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            details = data.get("details", {})
                            token_info = details.get("token_info", {})
                            
                            # Format token info
                            response_text = (
                                f"📊 *Detailed Token Information*\n\n"
                                f"*Name:* {token_info.get('name', 'Unknown')}\n"
                                f"*Symbol:* {token_info.get('symbol', 'Unknown')}\n"
                                f"*Decimals:* {token_info.get('decimals', 'Unknown')}\n"
                                f"*Total Supply:* {token_info.get('total_supply', 'Unknown'):,}\n"
                                f"*Creator:* {token_info.get('creator', 'Unknown')[:8]}...\n"
                                f"*Creation Time:* {token_info.get('creation_time', 'Unknown')}\n\n"
                            )
                            
                            # Add contract checks
                            checks = details.get("checks", {})
                            response_text += "*Contract Checks:*\n"
                            
                            for check, value in checks.items():
                                emoji = "✅" if not value else "⚠️"
                                if check in ["liquidity_locked"]:
                                    emoji = "✅" if value else "⚠️"
                                
                                # Format the check name for display
                                formatted_check = check.replace("_", " ").title()
                                
                                response_text += f"{emoji} {formatted_check}\n"
                            
                            await query.edit_message_text(
                                response_text,
                                parse_mode=ParseMode.MARKDOWN
                            )
                        else:
                            error_data = await response.json()
                            await query.edit_message_text(
                                f"❌ Error getting token info: {error_data.get('detail', 'Unknown error')}"
                            )
            except Exception as e:
                logger.error(f"Error handling button callback: {e}")
                await query.edit_message_text(f"❌ Error getting token info: {str(e)}")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages."""
        text = update.message.text
        user_id = update.effective_user.id
        
        # Extract potential token address if message contains one
        token_address_match = re.search(r'0x[a-fA-F0-9]{40}', text)
        if token_address_match:
            token_address = token_address_match.group(0)
            
            # Suggest security check
            keyboard = [
                [InlineKeyboardButton("Check Token Security", callback_data=f"check_token:{token_address}")]
            ]
            
            await update.message.reply_text(
                f"I noticed you shared a token address. Would you like to check its security?",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return
        
        # Extract potential trading pair if message contains one
        trading_pair_match = re.search(r'([A-Za-z0-9]+)/([A-Za-z0-9]+)', text)
        if trading_pair_match:
            symbol = trading_pair_match.group(0).upper()
            
            # Suggest price check
            keyboard = [
                [InlineKeyboardButton("Check Price", callback_data=f"check_price:{symbol}")]
            ]
            
            await update.message.reply_text(
                f"I noticed you mentioned {symbol}. Would you like to check its price?",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return
        
        # If no specific pattern is detected, provide general help
        await update.message.reply_text(
            "I'm your BROKR assistant. You can use commands like /price, /securitycheck, or /help to interact with me."
        )
    
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors in the dispatcher."""
        logger.error(f"Exception while handling update: {context.error}")
        
        # Get the user who caused the error
        if update and isinstance(update, Update) and update.effective_user:
            user_id = update.effective_user.id
            
            # Send error message to the user
            if update.effective_message:
                await update.effective_message.reply_text(
                    "Sorry, an error occurred while processing your request. Please try again later."
                )
    
    async def send_notification(self, chat_id: int, message: str, parse_mode: str = ParseMode.MARKDOWN, **kwargs):
        """Send a notification to a specific chat."""
        if not self.bot:
            logger.error("Bot not initialized. Cannot send notification.")
            return False
        
        try:
            await self.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=parse_mode,
                **kwargs
            )
            return True
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return False
    
    async def broadcast_to_authorized(self, message: str, parse_mode: str = ParseMode.MARKDOWN, **kwargs):
        """Broadcast a message to all authorized users."""
        if not self.bot:
            logger.error("Bot not initialized. Cannot broadcast message.")
            return False
        
        success_count = 0
        for user_id in self.authorized_users:
            try:
                await self.bot.send_message(
                    chat_id=user_id,
                    text=message,
                    parse_mode=parse_mode,
                    **kwargs
                )
                success_count += 1
            except Exception as e:
                logger.error(f"Error broadcasting to user {user_id}: {e}")
        
        return success_count

# Helper function to create and run the bot
def run_telegram_bot(token: str, api_url: str = "http://localhost:8000"):
    """Create and run the Telegram bot."""
    bot = BROKRTelegramBot(token, api_url)
    asyncio.run(bot.initialize())
    bot.start()

if __name__ == "__main__":
    # Example: Replace with your actual bot token from BotFather
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "your_bot_token_here")
    api_url = os.environ.get("BROKR_API_URL", "http://localhost:8000")
    
    run_telegram_bot(token, api_url)
