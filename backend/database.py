"""
Database module for AI Stock GPT
Handles PostgreSQL operations via SQLAlchemy for users, portfolios, chat history,
predictions, and email alerts.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from sqlalchemy import func, desc, distinct, text
from sqlalchemy.orm import joinedload

from .db_session import SessionLocal
from .models import (
    Base, User, Portfolio, Stock, ChatMessage, Prediction, EmailAlert,
    MarketData, SentimentScore, MacroIndicator, PasswordResetToken
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages PostgreSQL database operations via SQLAlchemy"""

    def __init__(self):
        """Initialize database session factory"""
        self._session_factory = SessionLocal
        logger.info("PostgreSQL DatabaseManager initialized")

    def ensure_admin_schema(self) -> None:
        """Add is_admin column on existing deployments (idempotent)."""
        session = self._get_session()
        try:
            session.execute(
                text("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT FALSE")
            )
            session.commit()
        except Exception as e:
            session.rollback()
            logger.warning(f"ensure_admin_schema: {e}")
        finally:
            session.close()

    def _get_session(self):
        return self._session_factory()

    # ------------------------------------------------------------------ #
    # User Management
    # ------------------------------------------------------------------ #

    def create_user(self, user_id: str, user_data: Dict[str, Any]) -> bool:
        """Create a new user"""
        session = self._get_session()
        try:
            user = User(
                id=user_id,
                email=user_data['email'],
                hashed_password=user_data['hashed_password'],
                first_name=user_data['first_name'],
                last_name=user_data['last_name'],
                username=user_data.get('username'),
                is_active=user_data.get('is_active', True),
                is_admin=user_data.get('is_admin', False),
            )
            session.add(user)
            session.commit()
            logger.info(f"User created successfully: {user_id}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to create user {user_id}: {e}")
            return False
        finally:
            session.close()

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user data by ID"""
        session = self._get_session()
        try:
            user = session.get(User, user_id)
            if user:
                return self._user_to_dict(user)
            return None
        except Exception as e:
            logger.error(f"Failed to get user {user_id}: {e}")
            return None
        finally:
            session.close()

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email address (case-insensitive)"""
        session = self._get_session()
        try:
            normalized = (email or "").strip().lower()
            user = session.query(User).filter(func.lower(User.email) == normalized).first()
            if user:
                result = self._user_to_dict(user)
                result['user_id'] = user.id
                return result
            return None
        except Exception as e:
            logger.error(f"Failed to get user by email {email}: {e}")
            return None
        finally:
            session.close()

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username"""
        session = self._get_session()
        try:
            user = session.query(User).filter(User.username == username).first()
            if user:
                result = self._user_to_dict(user)
                result['user_id'] = user.id
                return result
            return None
        except Exception as e:
            logger.error(f"Failed to get user by username {username}: {e}")
            return None
        finally:
            session.close()

    def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user data"""
        session = self._get_session()
        try:
            updates['updated_at'] = datetime.utcnow()
            session.query(User).filter(User.id == user_id).update(updates)
            session.commit()
            logger.info(f"User updated successfully: {user_id}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update user {user_id}: {e}")
            return False
        finally:
            session.close()

    def list_users(
        self,
        offset: int = 0,
        limit: int = 50,
        search: Optional[str] = None,
        include_inactive: bool = True,
    ) -> Dict[str, Any]:
        """List users for admin UI (paginated)."""
        session = self._get_session()
        try:
            q = session.query(User)
            if not include_inactive:
                q = q.filter(User.is_active.is_(True))
            if search:
                term = f"%{search.strip().lower()}%"
                q = q.filter(
                    (func.lower(User.email).like(term))
                    | (func.lower(User.first_name).like(term))
                    | (func.lower(User.last_name).like(term))
                    | (func.lower(User.username).like(term))
                )
            total = q.count()
            users = (
                q.order_by(desc(User.created_at))
                .offset(max(0, offset))
                .limit(min(200, max(1, limit)))
                .all()
            )
            return {
                "total": total,
                "offset": offset,
                "limit": limit,
                "users": [self.sanitize_user(self._user_to_dict(u)) for u in users],
            }
        except Exception as e:
            logger.error(f"list_users failed: {e}")
            return {"total": 0, "offset": offset, "limit": limit, "users": []}
        finally:
            session.close()

    def delete_user(self, user_id: str) -> bool:
        """Permanently delete a user and cascaded data."""
        session = self._get_session()
        try:
            user = session.get(User, user_id)
            if not user:
                return False
            session.delete(user)
            session.commit()
            logger.info(f"User deleted: {user_id}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"delete_user failed for {user_id}: {e}")
            return False
        finally:
            session.close()

    def get_user_admin_detail(self, user_id: str) -> Optional[Dict[str, Any]]:
        """User profile plus activity counts for admin."""
        session = self._get_session()
        try:
            user = session.get(User, user_id)
            if not user:
                return None
            chat_count = (
                session.query(func.count(ChatMessage.id))
                .filter(ChatMessage.user_id == user_id)
                .scalar()
                or 0
            )
            portfolio_count = (
                session.query(func.count(Portfolio.id))
                .filter(Portfolio.user_id == user_id)
                .scalar()
                or 0
            )
            prediction_count = (
                session.query(func.count(Prediction.id))
                .filter(Prediction.user_id == user_id)
                .scalar()
                or 0
            )
            detail = self.sanitize_user(self._user_to_dict(user))
            detail["stats"] = {
                "chat_messages": chat_count,
                "portfolios": portfolio_count,
                "predictions": prediction_count,
            }
            return detail
        except Exception as e:
            logger.error(f"get_user_admin_detail failed: {e}")
            return None
        finally:
            session.close()

    def promote_admin_by_emails(self, emails: List[str]) -> int:
        """Set is_admin=True for matching emails (bootstrap)."""
        if not emails:
            return 0
        session = self._get_session()
        try:
            normalized = [e.strip().lower() for e in emails if e.strip()]
            updated = (
                session.query(User)
                .filter(func.lower(User.email).in_(normalized))
                .update({"is_admin": True, "updated_at": datetime.utcnow()}, synchronize_session=False)
            )
            session.commit()
            return updated or 0
        except Exception as e:
            session.rollback()
            logger.error(f"promote_admin_by_emails failed: {e}")
            return 0
        finally:
            session.close()

    @staticmethod
    def sanitize_user(user: Dict[str, Any]) -> Dict[str, Any]:
        """Remove secrets before sending user dict to clients."""
        out = dict(user)
        out.pop("hashed_password", None)
        return out

    # ------------------------------------------------------------------ #
    # Password Reset Tokens
    # ------------------------------------------------------------------ #

    def create_password_reset_token(self, user_id: str, token: str, expires_at: datetime) -> bool:
        """Create a password reset token, invalidating any existing unused tokens for the user."""
        session = self._get_session()
        try:
            session.query(PasswordResetToken).filter(
                PasswordResetToken.user_id == user_id,
                PasswordResetToken.used == False
            ).update({"used": True})

            reset_token = PasswordResetToken(
                user_id=user_id,
                token=token,
                expires_at=expires_at,
                used=False,
            )
            session.add(reset_token)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to create password reset token: {e}")
            return False
        finally:
            session.close()

    def get_valid_reset_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Get a valid (unused, unexpired) password reset token."""
        session = self._get_session()
        try:
            reset_token = session.query(PasswordResetToken).filter(
                PasswordResetToken.token == token,
                PasswordResetToken.used == False,
                PasswordResetToken.expires_at > datetime.utcnow()
            ).first()
            if reset_token:
                return {
                    "id": str(reset_token.id),
                    "user_id": reset_token.user_id,
                    "token": reset_token.token,
                    "expires_at": reset_token.expires_at,
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get reset token: {e}")
            return None
        finally:
            session.close()

    def mark_reset_token_used(self, token: str) -> bool:
        """Mark a password reset token as used."""
        session = self._get_session()
        try:
            updated = session.query(PasswordResetToken).filter(
                PasswordResetToken.token == token
            ).update({"used": True})
            session.commit()
            return updated > 0
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to mark reset token as used: {e}")
            return False
        finally:
            session.close()

    # ------------------------------------------------------------------ #
    # Portfolio Management
    # ------------------------------------------------------------------ #

    def create_portfolio(self, user_id: str, portfolio_data_or_name=None, description=None) -> str:
        """Create a new portfolio for a user.

        Supports both dict form and positional args:
          create_portfolio(user_id, {"name": ..., "description": ...})
          create_portfolio(user_id, name, description)
        """
        session = self._get_session()
        try:
            if isinstance(portfolio_data_or_name, dict):
                name = portfolio_data_or_name.get('name', 'My Portfolio')
                desc = portfolio_data_or_name.get('description')
            else:
                name = portfolio_data_or_name or 'My Portfolio'
                desc = description

            portfolio = Portfolio(
                user_id=user_id,
                name=name,
                description=desc,
            )
            session.add(portfolio)
            session.commit()
            portfolio_id = str(portfolio.id)
            logger.info(f"Portfolio created for user {user_id}: {portfolio_id}")
            return portfolio_id
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to create portfolio for user {user_id}: {e}")
            return None
        finally:
            session.close()

    def get_user_portfolios(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all portfolios for a user"""
        session = self._get_session()
        try:
            portfolios = session.query(Portfolio).filter(
                Portfolio.user_id == user_id
            ).options(joinedload(Portfolio.stocks)).all()
            return [self._portfolio_to_dict(p) for p in portfolios]
        except Exception as e:
            logger.error(f"Failed to get portfolios for user {user_id}: {e}")
            return []
        finally:
            session.close()

    def add_stock_to_portfolio(self, user_id_or_portfolio_id: str,
                               portfolio_id_or_symbol: str = None,
                               stock_data_or_shares=None,
                               purchase_price: float = None,
                               purchase_date: str = None) -> bool:
        """Add a stock to a portfolio.

        Supports both original and enhanced call signatures:
          add_stock_to_portfolio(user_id, portfolio_id, {"symbol": ..., "shares": ...})
          add_stock_to_portfolio(portfolio_id, symbol, shares, purchase_price, purchase_date)
        """
        session = self._get_session()
        try:
            if isinstance(stock_data_or_shares, dict):
                # Original signature: (user_id, portfolio_id, stock_data_dict)
                portfolio_id = portfolio_id_or_symbol
                stock_data = stock_data_or_shares
                symbol = stock_data['symbol']
                shares = stock_data.get('shares', 0)
                price = stock_data.get('purchase_price', 0)
                date = stock_data.get('purchase_date')
            elif purchase_price is not None:
                # Enhanced signature: (portfolio_id, symbol, shares, purchase_price, purchase_date)
                portfolio_id = user_id_or_portfolio_id
                symbol = portfolio_id_or_symbol
                shares = stock_data_or_shares
                price = purchase_price
                date = purchase_date
            else:
                # Fallback: treat as (user_id, portfolio_id, stock_data_dict)
                portfolio_id = portfolio_id_or_symbol
                stock_data = stock_data_or_shares if isinstance(stock_data_or_shares, dict) else {}
                symbol = stock_data.get('symbol', '')
                shares = stock_data.get('shares', 0)
                price = stock_data.get('purchase_price', 0)
                date = stock_data.get('purchase_date')

            # Check if stock already exists in portfolio (upsert)
            existing = session.query(Stock).filter(
                Stock.portfolio_id == portfolio_id,
                Stock.symbol == symbol
            ).first()

            if existing:
                existing.shares = shares
                existing.purchase_price = price
                existing.purchase_date = date
                existing.updated_at = datetime.utcnow()
            else:
                stock = Stock(
                    portfolio_id=portfolio_id,
                    symbol=symbol,
                    shares=shares,
                    purchase_price=price,
                    purchase_date=date,
                )
                session.add(stock)

            session.commit()
            logger.info(f"Stock {symbol} added to portfolio {portfolio_id}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to add stock to portfolio: {e}")
            return False
        finally:
            session.close()

    def delete_stock_from_portfolio(self, portfolio_id: str, symbol: str) -> bool:
        """Delete a stock from a portfolio by symbol."""
        session = self._get_session()
        try:
            deleted = session.query(Stock).filter(
                Stock.portfolio_id == portfolio_id,
                Stock.symbol == symbol.upper()
            ).delete()
            session.commit()
            if deleted:
                logger.info(f"Stock {symbol} removed from portfolio {portfolio_id}")
            return deleted > 0
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to delete stock from portfolio: {e}")
            return False
        finally:
            session.close()

    def delete_portfolio(self, portfolio_id: str, user_id: str) -> bool:
        """Delete an entire portfolio (and its stocks via cascade) if it belongs to the user."""
        session = self._get_session()
        try:
            deleted = session.query(Portfolio).filter(
                Portfolio.id == portfolio_id,
                Portfolio.user_id == user_id
            ).delete()
            session.commit()
            if deleted:
                logger.info(f"Portfolio {portfolio_id} deleted for user {user_id}")
            return deleted > 0
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to delete portfolio: {e}")
            return False
        finally:
            session.close()

    def get_portfolio(self, user_id_or_portfolio_id: str, portfolio_id: str = None) -> Optional[Dict[str, Any]]:
        """Get portfolio data with stocks.

        Supports both signatures:
          get_portfolio(user_id, portfolio_id)
          get_portfolio(portfolio_id)
        """
        session = self._get_session()
        try:
            if portfolio_id is not None:
                # Original: (user_id, portfolio_id)
                pid = portfolio_id
            else:
                # Enhanced: (portfolio_id)
                pid = user_id_or_portfolio_id

            portfolio = session.query(Portfolio).filter(
                Portfolio.id == pid
            ).options(joinedload(Portfolio.stocks)).first()

            if portfolio:
                return self._portfolio_to_dict(portfolio)
            return None
        except Exception as e:
            logger.error(f"Failed to get portfolio: {e}")
            return None
        finally:
            session.close()

    # ------------------------------------------------------------------ #
    # Chat History
    # ------------------------------------------------------------------ #

    def save_chat_message(self, user_id: str, message_data_or_message=None, response: str = None) -> str:
        """Save a chat message to user's history.

        Supports both dict form and positional args:
          save_chat_message(user_id, {"message": ..., "response": ...})
          save_chat_message(user_id, message_str, response_str)
        """
        session = self._get_session()
        try:
            if isinstance(message_data_or_message, dict):
                message_text = message_data_or_message.get('message', '')
                response_text = message_data_or_message.get('response')
            else:
                message_text = str(message_data_or_message) if message_data_or_message else ''
                response_text = response

            msg = ChatMessage(
                user_id=user_id,
                message=message_text,
                response=response_text,
            )
            session.add(msg)
            session.commit()
            msg_id = str(msg.id)
            logger.info(f"Chat message saved for user {user_id}")
            return msg_id
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save chat message: {e}")
            return None
        finally:
            session.close()

    def get_chat_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get user's chat history"""
        session = self._get_session()
        try:
            messages = session.query(ChatMessage).filter(
                ChatMessage.user_id == user_id
            ).order_by(desc(ChatMessage.timestamp)).limit(limit).all()
            return [self._chat_to_dict(m) for m in messages]
        except Exception as e:
            logger.error(f"Failed to get chat history for user {user_id}: {e}")
            return []
        finally:
            session.close()

    # ------------------------------------------------------------------ #
    # Predictions and Analytics
    # ------------------------------------------------------------------ #

    def save_prediction(self, user_id: str, prediction_data_or_symbol=None,
                        prediction=None, days_ahead: int = None) -> str:
        """Save a stock prediction.

        Supports both dict form and positional args:
          save_prediction(user_id, {"symbol": ..., "prediction": ..., "days_ahead": ...})
          save_prediction(user_id, symbol, prediction, days_ahead)
        """
        session = self._get_session()
        try:
            if isinstance(prediction_data_or_symbol, dict):
                data = prediction_data_or_symbol
                symbol = data.get('symbol', '')
                pred_value = data.get('prediction', [])
                days = data.get('days_ahead', 5)
            else:
                symbol = prediction_data_or_symbol or ''
                pred_value = prediction
                days = days_ahead or 5

            # Convert numpy arrays to lists if needed
            if hasattr(pred_value, 'tolist'):
                pred_value = pred_value.tolist()
            if not isinstance(pred_value, list):
                pred_value = [pred_value] if pred_value is not None else []

            pred = Prediction(
                user_id=user_id,
                symbol=symbol,
                prediction=pred_value,
                days_ahead=days,
                status='pending',
            )
            session.add(pred)
            session.commit()
            pred_id = str(pred.id)
            logger.info(f"Prediction saved for user {user_id}")
            return pred_id
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save prediction: {e}")
            return None
        finally:
            session.close()

    def update_prediction_accuracy(self, prediction_id: str, actual_price: float, accuracy: float) -> bool:
        """Update prediction with actual results"""
        session = self._get_session()
        try:
            updated = session.query(Prediction).filter(
                Prediction.id == prediction_id
            ).update({
                'actual_price': actual_price,
                'accuracy': accuracy,
                'status': 'completed',
                'completed_at': datetime.utcnow(),
            })
            session.commit()
            if updated:
                logger.info(f"Prediction accuracy updated: {prediction_id}")
            return updated > 0
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update prediction accuracy: {e}")
            return False
        finally:
            session.close()

    # ------------------------------------------------------------------ #
    # Analytics and Reporting
    # ------------------------------------------------------------------ #

    def get_predictions_for_symbol(self, symbol: str, user_id: Optional[str] = None,
                                    start_date=None, end_date=None) -> List[Dict[str, Any]]:
        """Get stored predictions for a symbol, optionally filtered by user and date range."""
        session = self._get_session()
        try:
            query = session.query(Prediction).filter(Prediction.symbol == symbol.upper())
            if user_id:
                query = query.filter(Prediction.user_id == user_id)
            if start_date:
                query = query.filter(Prediction.created_at >= start_date)
            if end_date:
                query = query.filter(Prediction.created_at <= end_date)
            query = query.order_by(Prediction.created_at.desc()).limit(20)
            return [self._prediction_to_dict(p) for p in query.all()]
        except Exception as e:
            logger.error(f"Failed to get predictions for {symbol}: {e}")
            return []
        finally:
            session.close()

    def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user analytics"""
        session = self._get_session()
        try:
            analytics = {
                'total_predictions': 0,
                'accurate_predictions': 0,
                'average_accuracy': 0.0,
                'total_portfolio_value': 0.0,
                'total_gain_loss': 0.0,
                'favorite_stocks': [],
                'prediction_history': []
            }

            # Prediction stats
            completed = session.query(Prediction).filter(
                Prediction.user_id == user_id,
                Prediction.status == 'completed'
            ).all()

            analytics['total_predictions'] = len(completed)
            if completed:
                accurate = [p for p in completed if (p.accuracy or 0) > 0.7]
                analytics['accurate_predictions'] = len(accurate)
                analytics['average_accuracy'] = sum(p.accuracy or 0 for p in completed) / len(completed)
                analytics['prediction_history'] = [
                    self._prediction_to_dict(p) for p in completed[-10:]
                ]

            # Portfolio stats
            portfolio_stats = session.query(
                func.coalesce(func.sum(Portfolio.total_value), 0),
                func.coalesce(func.sum(Portfolio.total_gain_loss), 0)
            ).filter(Portfolio.user_id == user_id).first()

            analytics['total_portfolio_value'] = float(portfolio_stats[0])
            analytics['total_gain_loss'] = float(portfolio_stats[1])

            return analytics
        except Exception as e:
            logger.error(f"Failed to get user analytics for {user_id}: {e}")
            return {}
        finally:
            session.close()

    def get_system_analytics(self) -> Dict[str, Any]:
        """Get system-wide analytics"""
        session = self._get_session()
        try:
            analytics = {
                'total_users': 0,
                'total_predictions': 0,
                'average_prediction_accuracy': 0.0,
                'most_popular_stocks': [],
                'daily_active_users': 0
            }

            analytics['total_users'] = session.query(func.count(User.id)).scalar() or 0

            pred_stats = session.query(
                func.count(Prediction.id),
                func.avg(Prediction.accuracy)
            ).filter(Prediction.status == 'completed').first()

            analytics['total_predictions'] = pred_stats[0] or 0
            analytics['average_prediction_accuracy'] = float(pred_stats[1] or 0)

            yesterday = datetime.utcnow() - timedelta(days=1)
            dau = session.query(
                func.count(distinct(ChatMessage.user_id))
            ).filter(ChatMessage.timestamp >= yesterday).scalar()

            analytics['daily_active_users'] = dau or 0

            return analytics
        except Exception as e:
            logger.error(f"Failed to get system analytics: {e}")
            return {}
        finally:
            session.close()

    # ------------------------------------------------------------------ #
    # Email Alerts
    # ------------------------------------------------------------------ #

    def create_email_alert(self, user_id: str, symbol: str, alert_type: str,
                           threshold: float, email: str) -> Dict[str, Any]:
        """Create an email alert"""
        session = self._get_session()
        try:
            alert = EmailAlert(
                user_id=user_id,
                symbol=symbol,
                alert_type=alert_type,
                threshold=threshold,
                email=email,
            )
            session.add(alert)
            session.commit()
            result = self._alert_to_dict(alert)
            logger.info(f"Email alert created for user {user_id}: {symbol}")
            return result
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to create email alert: {e}")
            return None
        finally:
            session.close()

    def get_user_alerts(self, user_id: str) -> List[Dict[str, Any]]:
        """Get active email alerts for a user"""
        session = self._get_session()
        try:
            alerts = session.query(EmailAlert).filter(
                EmailAlert.user_id == user_id,
                EmailAlert.is_active == True
            ).all()
            return [self._alert_to_dict(a) for a in alerts]
        except Exception as e:
            logger.error(f"Failed to get user alerts: {e}")
            return []
        finally:
            session.close()

    # ------------------------------------------------------------------ #
    # Market Data
    # ------------------------------------------------------------------ #

    def save_market_data(self, symbol: str, date, ohlcv: Dict[str, Any], source: str) -> str:
        """Save or update market data (upsert pattern)."""
        session = self._get_session()
        try:
            existing = session.query(MarketData).filter(
                MarketData.symbol == symbol,
                MarketData.date == date,
                MarketData.source == source
            ).first()

            if existing:
                existing.open = ohlcv.get('open')
                existing.high = ohlcv.get('high')
                existing.low = ohlcv.get('low')
                existing.close = ohlcv.get('close')
                existing.volume = ohlcv.get('volume')
                existing.adjusted_close = ohlcv.get('adjusted_close')
                session.commit()
                return str(existing.id)
            else:
                md = MarketData(
                    symbol=symbol,
                    date=date,
                    open=ohlcv.get('open'),
                    high=ohlcv.get('high'),
                    low=ohlcv.get('low'),
                    close=ohlcv.get('close'),
                    volume=ohlcv.get('volume'),
                    adjusted_close=ohlcv.get('adjusted_close'),
                    source=source,
                )
                session.add(md)
                session.commit()
                return str(md.id)
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save market data for {symbol}: {e}")
            return None
        finally:
            session.close()

    def get_market_data(self, symbol: str, start_date, end_date) -> List[Dict[str, Any]]:
        """Get market data for a symbol within a date range."""
        session = self._get_session()
        try:
            rows = session.query(MarketData).filter(
                MarketData.symbol == symbol,
                MarketData.date >= start_date,
                MarketData.date <= end_date
            ).order_by(MarketData.date).all()
            return [
                {
                    'symbol': r.symbol, 'date': r.date,
                    'open': r.open, 'high': r.high, 'low': r.low,
                    'close': r.close, 'volume': r.volume,
                    'adjusted_close': r.adjusted_close, 'source': r.source,
                }
                for r in rows
            ]
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return []
        finally:
            session.close()

    def get_tracked_symbols(self) -> List[str]:
        """Get unique symbols from the stocks table."""
        session = self._get_session()
        try:
            symbols = session.query(distinct(Stock.symbol)).all()
            return [s[0] for s in symbols]
        except Exception as e:
            logger.error(f"Failed to get tracked symbols: {e}")
            return []
        finally:
            session.close()

    # ------------------------------------------------------------------ #
    # Sentiment
    # ------------------------------------------------------------------ #

    def save_sentiment_score(self, symbol: str, date, score: float,
                             label: str, confidence: float, source: str,
                             headline: str) -> str:
        """Save a sentiment score."""
        session = self._get_session()
        try:
            ss = SentimentScore(
                symbol=symbol,
                date=date,
                sentiment_score=score,
                sentiment_label=label,
                confidence=confidence,
                source=source,
                headline=headline,
            )
            session.add(ss)
            session.commit()
            return str(ss.id)
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save sentiment score for {symbol}: {e}")
            return None
        finally:
            session.close()

    def get_sentiment_scores(self, symbol: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get sentiment scores for a symbol over the past N days."""
        session = self._get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            rows = session.query(SentimentScore).filter(
                SentimentScore.symbol == symbol,
                SentimentScore.created_at >= cutoff
            ).order_by(desc(SentimentScore.date)).all()
            return [
                {
                    'symbol': r.symbol, 'date': r.date,
                    'sentiment_score': r.sentiment_score,
                    'sentiment_label': r.sentiment_label,
                    'confidence': r.confidence, 'source': r.source,
                    'headline': r.headline, 'created_at': r.created_at,
                }
                for r in rows
            ]
        except Exception as e:
            logger.error(f"Failed to get sentiment scores for {symbol}: {e}")
            return []
        finally:
            session.close()

    # ------------------------------------------------------------------ #
    # Macro Indicators
    # ------------------------------------------------------------------ #

    def save_macro_indicator(self, name: str, date, value: float,
                             source: str = 'fred') -> str:
        """Save or update a macro indicator (upsert pattern)."""
        session = self._get_session()
        try:
            existing = session.query(MacroIndicator).filter(
                MacroIndicator.indicator_name == name,
                MacroIndicator.date == date
            ).first()

            if existing:
                existing.value = value
                existing.source = source
                session.commit()
                return str(existing.id)
            else:
                mi = MacroIndicator(
                    indicator_name=name,
                    date=date,
                    value=value,
                    source=source,
                )
                session.add(mi)
                session.commit()
                return str(mi.id)
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save macro indicator {name}: {e}")
            return None
        finally:
            session.close()

    def get_macro_indicators(self, names: List[str], start_date, end_date) -> List[Dict[str, Any]]:
        """Get macro indicators by name within a date range."""
        session = self._get_session()
        try:
            rows = session.query(MacroIndicator).filter(
                MacroIndicator.indicator_name.in_(names),
                MacroIndicator.date >= start_date,
                MacroIndicator.date <= end_date
            ).order_by(MacroIndicator.date).all()
            return [
                {
                    'indicator_name': r.indicator_name, 'date': r.date,
                    'value': r.value, 'source': r.source,
                }
                for r in rows
            ]
        except Exception as e:
            logger.error(f"Failed to get macro indicators: {e}")
            return []
        finally:
            session.close()

    # ------------------------------------------------------------------ #
    # Helper methods for dict conversion
    # ------------------------------------------------------------------ #

    @staticmethod
    def _user_to_dict(user: User) -> Dict[str, Any]:
        return {
            'id': user.id,
            'user_id': user.id,
            'email': user.email,
            'hashed_password': user.hashed_password,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'username': user.username,
            'is_active': user.is_active,
            'is_admin': getattr(user, 'is_admin', False),
            'created_at': user.created_at,
            'updated_at': user.updated_at,
            'last_login': user.last_login,
        }

    @staticmethod
    def _portfolio_to_dict(portfolio: Portfolio) -> Dict[str, Any]:
        return {
            'id': str(portfolio.id),
            'user_id': portfolio.user_id,
            'name': portfolio.name,
            'description': portfolio.description,
            'total_value': portfolio.total_value,
            'total_gain_loss': portfolio.total_gain_loss,
            'created_at': portfolio.created_at,
            'updated_at': portfolio.updated_at,
            'stocks': [
                {
                    'symbol': s.symbol,
                    'shares': s.shares,
                    'purchase_price': s.purchase_price,
                    'purchase_date': str(s.purchase_date) if s.purchase_date else None,
                    'added_at': s.added_at,
                    'updated_at': s.updated_at,
                }
                for s in (portfolio.stocks or [])
            ],
        }

    @staticmethod
    def _chat_to_dict(msg: ChatMessage) -> Dict[str, Any]:
        return {
            'id': str(msg.id),
            'user_id': msg.user_id,
            'message': msg.message,
            'response': msg.response,
            'timestamp': msg.timestamp,
        }

    @staticmethod
    def _prediction_to_dict(pred: Prediction) -> Dict[str, Any]:
        return {
            'id': str(pred.id),
            'user_id': pred.user_id,
            'symbol': pred.symbol,
            'prediction': pred.prediction,
            'days_ahead': pred.days_ahead,
            'status': pred.status,
            'actual_price': pred.actual_price,
            'accuracy': pred.accuracy,
            'created_at': pred.created_at,
            'completed_at': pred.completed_at,
        }

    @staticmethod
    def _alert_to_dict(alert: EmailAlert) -> Dict[str, Any]:
        return {
            'id': str(alert.id),
            'user_id': alert.user_id,
            'symbol': alert.symbol,
            'alert_type': alert.alert_type,
            'threshold': alert.threshold,
            'email': alert.email,
            'is_active': alert.is_active,
            'created_at': alert.created_at,
            'triggered_at': alert.triggered_at,
        }


# Global database instance
db_manager = DatabaseManager()
