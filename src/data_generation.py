"""
data_generation.py  —  IT Helpdesk Ticket Classifier
=====================================================
Generates a seeded, reproducible 5,000-row synthetic helpdesk ticket dataset
with realistic class distributions, text variations, and resolution times.
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path

# ── seed ───────────────────────────────────────────────────────────────────────
GLOBAL_SEED = 42

# ── target distributions ───────────────────────────────────────────────────────
CATEGORY_WEIGHTS = {
    "Software":     0.25,
    "Network":      0.20,
    "Access/Login": 0.20,
    "Hardware":     0.15,
    "Email":        0.12,
    "Database":     0.08,
}

PRIORITY_BASE_WEIGHTS = {
    "Low":      0.35,
    "Medium":   0.40,
    "High":     0.18,
    "Critical": 0.07,
}

# ── SLA targets (used by routing engine) ──────────────────────────────────────
SLA_HOURS = {"Critical": 4, "High": 8, "Medium": 24, "Low": 72}

# ── resolution time parameters ────────────────────────────────────────────────
BASE_HOURS = {
    "Access/Login": 4,
    "Network":      6,
    "Email":        8,
    "Hardware":     12,
    "Software":     18,
    "Database":     24,
}
PRIORITY_MULTIPLIER = {"Critical": 0.5, "High": 1.0, "Medium": 2.0, "Low": 3.0}

# ═══════════════════════════════════════════════════════════════════════════════
#  COMPOSITIONAL TEXT BUILDING BLOCKS
#  Each ticket = subject + verb_phrase + obj + context + optional_tail
#  ~20 options per slot  →  20^4 > 160 000 unique combinations per category
# ═══════════════════════════════════════════════════════════════════════════════

# ── shared slot pools ─────────────────────────────────────────────────────────
_SUBJ = [
    "i", "my laptop", "our entire team", "several users on my floor",
    "me and two colleagues", "my workstation", "the new joiner",
    "everyone in the {team} department", "i and my manager",
    "all users in wing {wing}", "my desktop machine",
]
_SINCE = [
    "since this morning", "since yesterday afternoon", "after last night's update",
    "since the reboot", "from around 9 am today", "after the weekend",
    "since monday", "after the windows patch", "suddenly today",
    "since the power cut yesterday", "from this morning randomly",
]
_TAIL = [
    "please help asap", "kindly resolve this", "this is blocking my work",
    "need this fixed today", "please look into it", "urgent please",
    "any help appreciated", "thanks in advance", "please escalate if needed",
    "tried restarting but no luck", "already raised this verbally",
]
_TEAMS    = ["finance","HR","marketing","ops","sales","audit","legal","procurement"]
_WINGS    = ["B","C","D","east","west","ground floor"]
_SERVERS  = ["SRV-01","APPSRV-03","DC01","FSRV-02","SRV-DB01","WEBSVR-02"]
_SHARES   = ["Finance","HR","Projects","Reports","Invoices","Design"]
_NAMES    = ["Rahul","Priya","Amit","Sneha","Vikram","Divya","Rohan","Nisha","Karan"]
_SYSTEMS  = ["SAP","Jira","Confluence","Salesforce","ERP","CRM","intranet portal","Zoho"]
_DBS      = ["PROD_DB","ANALYTICS_DB","HR_DB","SALES_DB","ARCHIVE_DB","REPORTING_DB"]
_TABLES   = ["orders","employees","transactions","reports","audit_log","invoices","users"]
_APPS     = ["Zoom","Teams","Slack","Chrome","Excel","Tally","AutoCAD","Photoshop","Firefox"]
_LOCS     = ["conference room A","lab 3","reception area","wing B","cafeteria","server room"]
_CODES    = ["0x80070005","ERR_CONNECTION_RESET","0xc000021a","503","timeout","401","0x800f0954"]
_NS       = ["2","3","5","10","15","20","30","50","80","100"]
_PRINTERS = ["HP LaserJet","Canon printer","the shared printer","floor printer","reception printer"]

def _r(lst):
    return random.choice(lst)

def _fill(s):
    """Replace {key} tokens in string s with random slot values."""
    return (s
        .replace("{team}",   _r(_TEAMS))
        .replace("{wing}",   _r(_WINGS))
        .replace("{server}", _r(_SERVERS))
        .replace("{share}",  _r(_SHARES))
        .replace("{name}",   _r(_NAMES))
        .replace("{system}", _r(_SYSTEMS))
        .replace("{db}",     _r(_DBS))
        .replace("{table}",  _r(_TABLES))
        .replace("{app}",    _r(_APPS))
        .replace("{loc}",    _r(_LOCS))
        .replace("{code}",   _r(_CODES))
        .replace("{n}",      _r(_NS))
        .replace("{printer}",_r(_PRINTERS))
        .replace("{since}",  _r(_SINCE))
    )

# ── category-specific sentence cores ─────────────────────────────────────────
# Each template is a self-contained ticket description.
# 25-30 per category ensures template-level variety before slot-filling.
TEMPLATES = {

"Network": [
    "vpn not connecting {since}, error {code} keeps showing",
    "internet is completely down in {loc}, {n} users affected please help",
    "wifi keeps dropping every {n} minutes, very annoying",
    "cannot ping {server} from my laptop, 100 percent packet loss",
    "network share \\\\{server}\\{share} is inaccessible {since}",
    "internet speed extremely slow in {loc}, file uploads not completing",
    "vpn disconnects automatically after {n} minutes of idle time",
    "need {server} whitelisted on firewall, currently blocked",
    "network printer {printer} not showing on the network",
    "remote desktop to {server} fails with {code}, cannot access my work machine",
    "dns not resolving {server} from office laptops",
    "switch port in {loc} appears dead, no link light",
    "wifi password was reset and now {n} users in wing {wing} cannot connect",
    "site-to-site vpn tunnel keeps dropping, production systems unreachable",
    "load balancer marking {server} as down, half the application is broken",
    "connected to wifi but no internet access on my machine",
    "network share \\\\{server}\\{share} giving access denied to {team} team",
    "all printers offline on floor {n} after network maintenance last night",
    "intermittent packet loss to {server} causing application to freeze",
    "vpn client installer not working on windows 11, error {code}",
    "proxy settings blocking internal site on new laptop",
    "remote users cannot reach the office file server {since}",
    "network switch in {loc} rebooted itself, connectivity lost for {n} users",
    "wireless access point in {loc} not broadcasting ssid",
    "cannot access cloud portal from office network, works on mobile data",
    "latency spikes to {server} causing database timeouts",
],

"Hardware": [
    "laptop screen flickering and sometimes goes completely black",
    "spacebar and enter key sticking on my keyboard",
    "computer not turning on, no lights no fan no response",
    "mouse pointer freezing every few minutes, tried different usb port",
    "paper jam in {printer} in {loc}, cleared it but keeps happening",
    "docking station not detecting second monitor since last week",
    "laptop battery drains in {n} hours even when plugged in",
    "hard drive making clicking noise, afraid of losing data",
    "desktop fan very loud, machine shutting down due to heat",
    "front usb ports on desktop not working at all",
    "hdmi monitor shows no signal after sleep mode",
    "laptop randomly shuts off during meetings, very disruptive",
    "scanner in {loc} not detected after driver update",
    "projector in {loc} not displaying, important presentation in {n} mins",
    "numpad not working after coffee spill on keyboard yesterday",
    "headphone jack on laptop broken, needed for client calls",
    "laptop charger cable fraying, worried it will stop working",
    "desktop monitor has dead pixels in the center of screen",
    "barcode scanner at reception stopped working {since}",
    "webcam not detected in teams and zoom on my laptop",
    "laptop hinge is broken, screen wobbles during calls",
    "ram seems low, machine very slow when opening {n} tabs",
    "ethernet port on laptop loose, cable keeps falling out",
    "office phone handset not working, no dial tone",
    "touch screen on surface pro not responding to input",
    "portable hard disk not recognised when plugged into my laptop",
],

"Software": [
    "outlook keeps crashing when i open attachments {since}",
    "excel file says corrupted and wont open, need data urgently",
    "{app} crashes immediately on startup, reinstalling didnt help",
    "zoom freezes after {n} minutes on video calls",
    "teams stuck on grey loading screen, cannot join standup",
    "software installation failing with error {code}",
    "windows update stuck at {n} percent for {n} hours now",
    "chrome not launching at all, task manager shows it running",
    "antivirus quarantined {app} exe, need it whitelisted",
    "word document not saving, losing work repeatedly",
    "{app} throws error {code} on every launch since the update",
    "screen resolution changed after windows update, everything looks blurry",
    "{app} using 99 percent cpu making laptop unusable",
    "onedrive not syncing, files stuck in pending {since}",
    "python import error for numpy on my work machine",
    "calculator and snipping tool missing after windows 11 upgrade",
    "all browser bookmarks disappeared after chrome update",
    "machine boots into safe mode randomly, very slow to use",
    "{app} license expired popup appearing every few minutes",
    "cannot install {app} as it says another version already installed",
    "pdf files not opening with adobe, asks to associate every time",
    "right click context menu extremely slow to appear on desktop",
    "start menu search not working after cumulative update",
    "vpn software and antivirus conflicting, one crashes the other",
    "task scheduler job failing with {code} on {server}",
    "software upgrade broke integration with {system}, urgent",
    "audio driver missing after windows update, no sound",
    "dual monitor setup broke after driver update, only one screen works",
],

"Access/Login": [
    "locked out of my account after {n} failed attempts {since}",
    "need access to {system} for {team} team, approved by manager",
    "password reset email not arriving, already checked spam",
    "vpn credentials being rejected even after password reset",
    "need shared drive \\\\{server}\\{share} access for {team} project",
    "new joiner {name} needs laptop setup and {system} access today",
    "account auto-locking every morning, happens around 9 am daily",
    "need temporary local admin rights to install {app}",
    "otp for two-factor auth not arriving on my registered number",
    "colleague {name} left, please transfer their {share} folder access to me",
    "sso login to {system} failing with error {code}",
    "forgot windows password and machine is now locked",
    "need read-only access to {system} for upcoming audit by {team} team",
    "cannot access company portal from home, works in office",
    "fingerprint login broken after windows update, have to type password",
    "new employee starting monday, need {system} and email access ready",
    "vendor {name} needs temporary guest wifi and {system} access for 2 days",
    "role changed to {team} manager, need additional permissions in {system}",
    "shared mailbox access for {team} not working after mailbox migration",
    "account disabled by mistake, cannot login to anything",
    "need to reset mfa, lost access to authenticator app after phone change",
    "contractor needs access to {system} extended by {n} weeks",
    "service account password expired, {app} stopped working",
    "need bulk access provisioning for {n} new joiners in {team} team",
    "privileged access request for {server} for weekend maintenance",
    "user {name} not able to access {share} after folder permission change",
],

"Email": [
    "outlook not receiving any new emails {since}",
    "sent emails stuck in outbox and not delivering",
    "calendar invite not appearing in other persons calendar",
    "getting spoofed spam from what looks like internal addresses",
    "email signature disappeared after recent outlook update",
    "inbox rules stopped working, emails landing in wrong folders",
    "email attachments not downloading, just shows spinning wheel",
    "shared mailbox for {team} not loading in outlook",
    "emails bouncing back with error {code}",
    "out of office message not activating despite being set correctly",
    "emails taking {n} hours to deliver even within the company",
    "cannot send emails over {n} mb, getting rejected by server",
    "outlook prompting for password every few minutes",
    "mobile phone not syncing emails and calendar with outlook",
    "meeting invites showing wrong time zone for recipients",
    "global address list not showing new employees added this week",
    "emails from {name} going to junk even after whitelist",
    "outlook extremely slow when opening, takes {n} minutes to load",
    "cannot access old archived emails, search returns nothing",
    "distribution list {team} not delivering to all members",
    "email encryption certificate expired, cannot send secure emails",
    "shared calendar for {team} team not visible to new members",
    "bulk email to {team} bounced for {n} recipients",
    "outlook desktop and web showing different inbox contents",
    "email recall feature not working, message already read",
],

"Database": [
    "sql query timing out after {n} seconds on {server}",
    "{db} not accepting new connections {since}",
    "stored procedure in {db} throwing {code} since deployment",
    "scheduled backup failed last night for {db}",
    "table {table} in {db} is locked, blocking all other queries",
    "need read access to {db} for {team} reporting, manager approved",
    "replication lag between primary and standby on {server} is {n} mins",
    "{db} storage 95 percent full, need expansion urgently",
    "indexes on {table} fragmented, all queries running very slow",
    "production {db} at {n} percent cpu, application is crawling",
    "transaction log on {db} is full, causing write failures",
    "application cannot connect to {db} from {server} {since}",
    "deadlock occurring between two jobs on {table} every {n} mins",
    "need to restore {db} to yesterday point in time for data issue",
    "wrong data committed to {table}, need rollback assistance",
    "statistics on {db} outdated, query plans are inefficient",
    "database link between {server} and {db} broken after network change",
    "need dump of {table} from {db} for audit by {team} team",
    "connection pool exhausted on {server}, app returning errors",
    "nosql cache out of sync with {db}, stale data being served",
    "database certificate expiring in {n} days, renewal needed",
    "temp tablespace on {db} filling up during month-end reports",
],

}

# ── urgency phrases for High/Critical ─────────────────────────────────────────
URGENCY_PREFIX = [
    "urgent - ", "production is down - ", "critical issue - ",
    "please help immediately, ", "whole team blocked - ",
    "business impacted - ", "escalating this - ",
    "cannot work at all, ", "everyone affected - ",
]

LOW_PREFIX = [
    "no rush but ", "whenever you get a chance, ",
    "low priority - ", "at your convenience, ", "not urgent - ",
]

# ── typo injection ────────────────────────────────────────────────────────────
def _inject_typo(word: str) -> str:
    """Apply one random character-level typo (5 % probability per word)."""
    if len(word) < 3 or random.random() > 0.05:
        return word
    i = random.randint(0, len(word) - 2)
    op = random.randint(0, 2)
    if op == 0:                                    # swap adjacent chars
        lst = list(word); lst[i], lst[i+1] = lst[i+1], lst[i]; return "".join(lst)
    elif op == 1: return word[:i] + word[i+1:]     # drop char
    else:         return word[:i] + word[i]*2 + word[i+1:]  # duplicate char

def _apply_typos(text: str) -> str:
    return " ".join(_inject_typo(w) for w in text.split())

# ── Jaccard similarity guard ───────────────────────────────────────────────────
def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

def _is_near_duplicate(text: str, existing: list, threshold: float = 0.85,
                        sample_size: int = 50) -> bool:
    """Check text against a random sample of existing texts."""
    pool = existing[-sample_size:] if len(existing) > sample_size else existing
    return any(_jaccard(text, e) >= threshold for e in pool)

# ── main text generator ───────────────────────────────────────────────────────
def _generate_text(category: str, priority: str,
                   existing: list, max_retries: int = 8) -> str:
    """
    Build a ticket text for (category, priority).
    Tries up to max_retries times to avoid near-duplicates of existing texts.
    """
    for _ in range(max_retries):
        # pick template, fill slots
        raw  = _fill(random.choice(TEMPLATES[category]))

        # prepend optional subject clause (30 % chance)
        if random.random() < 0.30:
            subj = _fill(random.choice(_SUBJ))
            raw  = f"{subj} - {raw}"

        # prepend priority keyword phrase
        if priority in ("High", "Critical") and random.random() < 0.70:
            raw = random.choice(URGENCY_PREFIX) + raw
        elif priority == "Low" and random.random() < 0.40:
            raw = random.choice(LOW_PREFIX) + raw

        # append tail (40 % chance)
        if random.random() < 0.40:
            raw = raw + ", " + random.choice(_TAIL)

        text = _apply_typos(raw)
        if not _is_near_duplicate(text, existing):
            return text

    # fallback: return last attempt even if similar (very rare)
    return text

# ── timestamp generation ──────────────────────────────────────────────────────
def generate_timestamps(n_rows: int, seed: int = GLOBAL_SEED) -> pd.Series:
    """
    Sample n_rows datetime values from a 90-day window.
    Monday/Tuesday are 2× more likely (post-weekend backlog).
    Business hours (08-18) are 3× more likely than off-hours.
    """
    rng   = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-08")
    end   = pd.Timestamp("2024-04-07")
    mins  = pd.date_range(start, end, freq="min")

    dow_w  = np.where(mins.dayofweek < 2, 2.0, 1.0)
    biz_w  = np.where((mins.hour >= 8) & (mins.hour < 18), 3.0, 1.0)
    prob   = (dow_w * biz_w); prob /= prob.sum()

    idx = rng.choice(len(mins), size=n_rows, replace=True, p=prob)
    return pd.Series(mins[idx]).reset_index(drop=True)

# ── resolution time ───────────────────────────────────────────────────────────
def compute_resolution_time(category: str, priority: str,
                             rng: np.random.Generator) -> float:
    """
    resolution_hours = base[category] × multiplier[priority] × lognormal_noise
    Lognormal keeps values positive and adds a realistic fat right tail.
    """
    base  = BASE_HOURS[category]
    mult  = PRIORITY_MULTIPLIER[priority]
    noise = rng.lognormal(mean=0.0, sigma=0.4)
    return round(max(0.5, base * mult * noise), 2)

# ── master generation function ────────────────────────────────────────────────
def generate_dataset(n: int = 5000, seed: int = GLOBAL_SEED) -> pd.DataFrame:
    """
    Generate the full synthetic helpdesk ticket DataFrame.

    Parameters
    ----------
    n    : number of rows (default 5 000)
    seed : random seed — change this to get a different but reproducible dataset

    Returns
    -------
    pd.DataFrame  (7 columns, no nulls)
    """
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    categories = list(CATEGORY_WEIGHTS.keys())
    cat_w      = list(CATEGORY_WEIGHTS.values())
    priorities = list(PRIORITY_BASE_WEIGHTS.keys())
    pri_w      = list(PRIORITY_BASE_WEIGHTS.values())

    ticket_texts, cats, pris, res_times = [], [], [], []

    for _ in range(n):
        cat  = random.choices(categories, weights=cat_w,  k=1)[0]
        pri  = random.choices(priorities, weights=pri_w,  k=1)[0]
        text = _generate_text(cat, pri, ticket_texts)
        res  = compute_resolution_time(cat, pri, rng)

        ticket_texts.append(text)
        cats.append(cat)
        pris.append(pri)
        res_times.append(res)

    df = pd.DataFrame({
        "ticket_id":             [f"TKT-{i:04d}" for i in range(1, n + 1)],
        "ticket_text":           ticket_texts,
        "category":              cats,
        "priority":              pris,
        "created_at":            generate_timestamps(n, seed),
        "resolution_time_hours": res_times,
    })
    df["resolved_at"] = df["created_at"] + pd.to_timedelta(
        df["resolution_time_hours"], unit="h"
    )

    # final column order matching spec
    return df[["ticket_id","ticket_text","category","priority",
               "created_at","resolution_time_hours","resolved_at"]]


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    out = Path("data/raw/tickets.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating dataset  n=5000  seed={GLOBAL_SEED} ...")
    df = generate_dataset()
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows -> {out}")
    print("\nCategory distribution:")
    print(df["category"].value_counts())
    print("\nPriority distribution:")
    print(df["priority"].value_counts())
    print("\nResolution time (hours) summary:")
    print(df["resolution_time_hours"].describe().round(2))
