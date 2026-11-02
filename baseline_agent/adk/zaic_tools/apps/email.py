"""email: one method per tool of the app. Inbox in emails[] (folder inbox/archive), outgoing mail in sent[], drafts in drafts[]."""
from __future__ import annotations

import logging
import re

from google.adk.tools import ToolContext

from .base import App, read, write

logger = logging.getLogger(__name__)

_ADDR = re.compile(r"^[^@\s<>]+@[^@\s<>]+\.[^@\s<>]+$")


def _addresses(v, field):
    """(list, None) when v is a list of e-mail addresses, else (None, message). Presence/type is the catalog's job."""
    if v is None:
        return [], None
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list):
        v = [v]
    bad = [a for a in v if not isinstance(a, str) or not _ADDR.match(a.strip())]
    if bad:
        return None, f"{field}: địa chỉ email không hợp lệ {bad[0]!r}"
    return [a.strip() for a in v], None


class Email(App):
    name = "email"

    @read
    def list(self, unread_only=None, **_):
        """
        This tool lists emails in the user's inbox or archive.
        Args:
            unread_only (bool): If True, filters and returns only unread emails. Default to None
        Returns:
            dict: A dictionary containing the list of emails (excluding bodies).
        """
        logger.info("Email.list invoked")
        r = App.read(self, "emails", drop=("body",))
        if unread_only:
            r["emails"] = [e for e in r["emails"] if e.get("unread")]
        return r

    @read
    def read(self, email_id=None, **_):
        """
        This tool reads the full details of a specific email by its ID.
        Args:
            email_id (str): The unique identifier of the email to read. Default to None
        Returns:
            dict: A dictionary containing the email details or an error message if not found.
        """
        logger.info("Email.read invoked")
        e = self.find("emails", email_id)
        if e is None:
            return self.noop(f"không tìm thấy email_id={email_id!r}")
        return {"ok": True, "email": dict(e)}

    def _outgoing(self, collection, prefix, to, subject, body, cc):
        to, err = _addresses(to, "to")
        if err:
            return self.noop(err)
        cc, err = _addresses(cc, "cc")
        if err:
            return self.noop(err)
        r = App.create(self, collection, prefix, to=to, cc=cc, subject=subject, body=body)
        return {"ok": True, "id": r["id"], collection[:-1] if collection == "drafts" else "email": r["item"]}

    @write
    def send(self, to=None, subject=None, body=None, cc=None, **_):
        """
        This tool sends an email to one or more recipients.
        Args:
            to (str | list[str]): The recipient email address or list of addresses. Default to None
            subject (str): The subject line of the email. Default to None
            body (str): The main content/body of the email. Default to None
            cc (str | list[str]): Carbon copy email address or list of addresses. Default to None
        Returns:
            dict: A dictionary containing the status, sent email ID, and email item details.
        """
        logger.info("Email.send invoked")
        return self._outgoing("sent", "sent", to, subject, body, cc)

    @write
    def create_draft(self, to=None, subject=None, body=None, cc=None, **_):
        """ 
            This tool creates an email draft for user.
            Args: 
                to (str | list[str]): A string or list of email addresses trying to send to. Default to None
                subject (str): A subject of a draft email. Default to None
                body (str): The content of an email. Default to None
                cc (str | list[str]): A list or string of emails that the user referring to in this email. Default to None
            Returns:
                dict: A dictionary containing the status, draft ID, and draft item details.
        """
        logger.info("Email.create_draft invoked")
        return self._outgoing("drafts", "draft", to, subject, body, cc)

    @write
    def reply(self, email_id=None, body=None, **_):
        """
        This tool replies to an existing email.
        Args:
            email_id (str): The unique identifier of the email being replied to. Default to None
            body (str): The reply content/body. Default to None
        Returns:
            dict: A dictionary containing the status, sent reply ID, and email item details.
        """
        logger.info("Email.reply invoked")
        e = self.find("emails", email_id)
        if e is None:
            return self.noop(f"không tìm thấy email_id={email_id!r}")
        subj = e.get("subject") or ""
        r = App.create(self, "sent", "sent", to=[e.get("from")], cc=[], subject=subj if subj.lower().startswith("re:") else f"Re: {subj}", body=body, in_reply_to=e["id"])
        e["unread"] = False
        e["replied"] = True
        return {"ok": True, "id": r["id"], "email": r["item"]}

    @write
    def archive(self, email_id=None, action=None, **_):
        """
        This tool archives or deletes a specific email.
        Args:
            email_id (str): The unique identifier of the email to archive or delete. Default to None
            action (str): The action to perform ("delete" to permanently delete, otherwise archives). Default to None
        Returns:
            dict: A dictionary containing the operation result status.
        """
        logger.info("Email.archive invoked")
        if action == "delete":
            return App.delete(self, "emails", "email_id", email_id)
        return self.patch("emails", "email_id", email_id, set={"folder": "archive"})