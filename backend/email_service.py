"""
Email service for AI Stock GPT
Uses SendGrid to send transactional emails (password reset, etc.)
"""

import os
import logging

logger = logging.getLogger(__name__)

SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", "noreply@aistockgpt.com")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")


class EmailService:
    """Handles sending transactional emails via SendGrid"""

    def __init__(self):
        self.api_key = SENDGRID_API_KEY
        self.from_email = FROM_EMAIL
        self.frontend_url = FRONTEND_URL
        self.client = None

        if self.api_key and self.api_key != "your-sendgrid-api-key-here":
            try:
                from sendgrid import SendGridAPIClient
                self.client = SendGridAPIClient(self.api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize SendGrid client: {e}")

    def send_password_reset_email(self, to_email: str, reset_token: str, first_name: str) -> bool:
        """Send password reset email with a reset link."""
        reset_link = f"{self.frontend_url}/reset-password?token={reset_token}"

        if not self.client:
            logger.warning("SendGrid not configured — logging reset link for dev use")
            logger.info(f"Password reset link for {to_email}: {reset_link}")
            return True

        try:
            from sendgrid.helpers.mail import Mail, Email, To, HtmlContent

            html_content = f"""
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="text-align: center; margin-bottom: 30px;">
                    <h1 style="color: #6366f1; margin: 0;">AI Stock GPT</h1>
                    <p style="color: #6b7280; margin-top: 4px;">Intelligent Stock Analysis</p>
                </div>

                <div style="background: #f9fafb; border-radius: 12px; padding: 30px; border: 1px solid #e5e7eb;">
                    <h2 style="color: #111827; margin-top: 0;">Password Reset Request</h2>
                    <p style="color: #374151;">Hi {first_name},</p>
                    <p style="color: #374151;">
                        We received a request to reset your password. Click the button below
                        to set a new password. This link will expire in 30 minutes.
                    </p>

                    <div style="text-align: center; margin: 30px 0;">
                        <a href="{reset_link}"
                           style="background: linear-gradient(to right, #6366f1, #7c3aed);
                                  color: white; padding: 14px 32px; border-radius: 8px;
                                  text-decoration: none; font-weight: 600; font-size: 16px;
                                  display: inline-block;">
                            Reset Password
                        </a>
                    </div>

                    <p style="color: #6b7280; font-size: 13px;">
                        If you didn't request this, you can safely ignore this email.
                        Your password will remain unchanged.
                    </p>

                    <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 20px 0;">

                    <p style="color: #9ca3af; font-size: 12px;">
                        If the button doesn't work, copy and paste this link into your browser:<br>
                        <a href="{reset_link}" style="color: #6366f1; word-break: break-all;">{reset_link}</a>
                    </p>
                </div>

                <p style="color: #9ca3af; font-size: 11px; text-align: center; margin-top: 20px;">
                    &copy; AI Stock GPT. This is an automated message, please do not reply.
                </p>
            </div>
            """

            message = Mail(
                from_email=Email(self.from_email, "AI Stock GPT"),
                to_emails=To(to_email),
                subject="Reset Your AI Stock GPT Password",
                html_content=HtmlContent(html_content),
            )

            response = self.client.send(message)
            logger.info(f"Password reset email sent to {to_email}, status: {response.status_code}")
            return response.status_code in (200, 201, 202)

        except Exception as e:
            logger.error(f"Failed to send password reset email to {to_email}: {e}")
            return False


# Global email service instance
email_service = EmailService()
