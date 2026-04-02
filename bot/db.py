"""
db.py — SQLite trade database for analytics and historical queries.

Stores every trade with full context: FVG data, strategy cell, fill quality,
commissions, slippage, duration. Queryable for performance analysis.

JSONL logs are for real-time streaming/debugging.
SQLite is for analytics: "which cells make money?", "what's my avg slippage?",
"profit factor by setup type this month?", etc.
"""

import os
import sqlite3
from datetime import datetime


DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "bot_trades.db"
)


def get_connection(db_path=None):
    """Get a SQLite connection with WAL mode for concurrent reads."""
    path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=None):
    """Create tables and indexes. Safe to call multiple times."""
    conn = get_connection(db_path)
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            -- Identifiers
            group_id TEXT UNIQUE NOT NULL,
            fvg_id TEXT NOT NULL,
            trade_date TEXT NOT NULL,

            -- FVG context
            fvg_type TEXT NOT NULL,
            zone_low REAL,
            zone_high REAL,
            fvg_size REAL,
            time_period TEXT NOT NULL,
            risk_range TEXT NOT NULL,

            -- Strategy cell
            setup TEXT NOT NULL,
            n_value REAL NOT NULL,
            cell_ev REAL,
            cell_win_rate REAL,

            -- Order details
            side TEXT NOT NULL,
            contracts INTEGER NOT NULL,
            entry_price REAL NOT NULL,
            stop_price REAL NOT NULL,
            target_price REAL NOT NULL,
            risk_pts REAL NOT NULL,

            -- Fill quality
            actual_entry_price REAL,
            entry_slippage_pts REAL DEFAULT 0,
            actual_exit_price REAL,
            stop_slippage_pts REAL DEFAULT 0,

            -- Outcome
            exit_reason TEXT,
            pnl_pts REAL DEFAULT 0,
            gross_pnl REAL DEFAULT 0,
            commission REAL DEFAULT 0,
            net_pnl REAL DEFAULT 0,

            -- Timing
            entry_time TEXT,
            exit_time TEXT,
            duration_seconds INTEGER,

            -- Account context
            balance_before REAL,
            balance_after REAL,
            daily_pnl_after REAL,

            -- Metadata
            strategy_id TEXT,
            mode TEXT DEFAULT 'PAPER',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS daily_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date TEXT UNIQUE NOT NULL,

            start_balance REAL,
            end_balance REAL,
            net_pnl REAL,
            pnl_pct REAL,

            total_trades INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            eod_exits INTEGER DEFAULT 0,
            win_rate REAL DEFAULT 0,

            gross_profit REAL DEFAULT 0,
            gross_loss REAL DEFAULT 0,
            profit_factor REAL,
            avg_win REAL DEFAULT 0,
            avg_loss REAL DEFAULT 0,

            total_contracts INTEGER DEFAULT 0,
            total_commission REAL DEFAULT 0,
            total_slippage_pts REAL DEFAULT 0,

            fvgs_detected INTEGER DEFAULT 0,
            fvgs_mitigated INTEGER DEFAULT 0,
            setups_rejected INTEGER DEFAULT 0,
            setups_accepted INTEGER DEFAULT 0,

            max_drawdown REAL DEFAULT 0,
            kill_switch_hit INTEGER DEFAULT 0,
            strategy_id TEXT,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS fvg_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fvg_id TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            fvg_type TEXT NOT NULL,
            zone_low REAL,
            zone_high REAL,
            fvg_size REAL,
            time_period TEXT,
            formation_time TEXT,
            mitigated INTEGER DEFAULT 0,
            mitigation_time TEXT,
            trade_placed INTEGER DEFAULT 0,
            rejection_reason TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            message TEXT NOT NULL,
            sent INTEGER DEFAULT 0,
            send_attempts INTEGER DEFAULT 0,
            last_error TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            sent_at TEXT
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS reconciliation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date TEXT UNIQUE NOT NULL,
            live_count INTEGER,
            backtest_count INTEGER,
            matched_count INTEGER,
            divergence_count INTEGER,
            live_net_pnl REAL,
            backtest_net_pnl REAL,
            divergences_json TEXT,
            error TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Indexes for common queries
    c.execute('CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(trade_date)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_trades_setup ON trades(setup)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_trades_time_period ON trades(time_period)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_trades_exit_reason ON trades(exit_reason)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_fvg_log_date ON fvg_log(trade_date)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_daily_stats_date ON daily_stats(trade_date)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_recon_date ON reconciliation_results(trade_date)')

    conn.commit()
    conn.close()


class TradeDB:
    """
    Interface for reading/writing trade data to SQLite.

    Usage:
        db = TradeDB()
        db.insert_trade(group_id="abc", ...)
        db.update_trade_exit(group_id="abc", exit_reason="TP", ...)
        stats = db.query_cell_performance("10:30-11:00", "10-15")
    """

    def __init__(self, db_path=None):
        self._db_path = db_path or DEFAULT_DB_PATH
        init_db(self._db_path)

    def _conn(self):
        return get_connection(self._db_path)

    # ── Writes ────────────────────────────────────────────────────────────

    def insert_trade(self, **kwargs):
        """Insert a new trade when order is placed."""
        conn = self._conn()
        cols = ', '.join(kwargs.keys())
        placeholders = ', '.join(['?'] * len(kwargs))
        conn.execute(
            f'INSERT OR REPLACE INTO trades ({cols}) VALUES ({placeholders})',
            list(kwargs.values())
        )
        conn.commit()
        conn.close()

    def update_trade_exit(self, group_id, **kwargs):
        """Update a trade with exit data (fill price, P&L, commission, etc.)."""
        conn = self._conn()
        sets = ', '.join(f'{k} = ?' for k in kwargs.keys())
        conn.execute(
            f'UPDATE trades SET {sets} WHERE group_id = ?',
            list(kwargs.values()) + [group_id]
        )
        conn.commit()
        conn.close()

    def insert_daily_stats(self, **kwargs):
        """Insert or update daily stats."""
        conn = self._conn()
        cols = ', '.join(kwargs.keys())
        placeholders = ', '.join(['?'] * len(kwargs))
        conflict_sets = ', '.join(f'{k} = excluded.{k}' for k in kwargs.keys() if k != 'trade_date')
        conn.execute(
            f'INSERT INTO daily_stats ({cols}) VALUES ({placeholders}) '
            f'ON CONFLICT(trade_date) DO UPDATE SET {conflict_sets}',
            list(kwargs.values())
        )
        conn.commit()
        conn.close()

    def insert_reconciliation(self, **kwargs):
        """Insert or update reconciliation result for a trading day."""
        conn = self._conn()
        cols = ', '.join(kwargs.keys())
        placeholders = ', '.join(['?'] * len(kwargs))
        conflict_sets = ', '.join(f'{k} = excluded.{k}' for k in kwargs.keys() if k != 'trade_date')
        conn.execute(
            f'INSERT INTO reconciliation_results ({cols}) VALUES ({placeholders}) '
            f'ON CONFLICT(trade_date) DO UPDATE SET {conflict_sets}',
            list(kwargs.values())
        )
        conn.commit()
        conn.close()

    def insert_fvg(self, **kwargs):
        """Log an FVG detection event."""
        conn = self._conn()
        cols = ', '.join(kwargs.keys())
        placeholders = ', '.join(['?'] * len(kwargs))
        conn.execute(
            f'INSERT INTO fvg_log ({cols}) VALUES ({placeholders})',
            list(kwargs.values())
        )
        conn.commit()
        conn.close()

    def update_fvg(self, fvg_id, **kwargs):
        """Update an FVG record (e.g., mark as mitigated)."""
        conn = self._conn()
        sets = ', '.join(f'{k} = ?' for k in kwargs.keys())
        conn.execute(
            f'UPDATE fvg_log SET {sets} WHERE fvg_id = ?',
            list(kwargs.values()) + [fvg_id]
        )
        conn.commit()
        conn.close()

    # ── Queries ───────────────────────────────────────────────────────────

    def query(self, sql, params=None):
        """Run a raw SQL query and return list of dicts."""
        conn = self._conn()
        rows = conn.execute(sql, params or []).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_trades(self, date=None, setup=None, limit=100):
        """Get trades with optional filters."""
        sql = 'SELECT * FROM trades WHERE 1=1'
        params = []
        if date:
            sql += ' AND trade_date = ?'
            params.append(date)
        if setup:
            sql += ' AND setup = ?'
            params.append(setup)
        sql += ' ORDER BY entry_time DESC LIMIT ?'
        params.append(limit)
        return self.query(sql, params)

    def get_cell_performance(self, time_period=None, risk_range=None,
                             setup=None, days=None):
        """Aggregate P&L by strategy cell."""
        sql = '''
            SELECT
                time_period, risk_range, setup, n_value,
                COUNT(*) as trades,
                SUM(CASE WHEN exit_reason = 'TP' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN exit_reason = 'SL' THEN 1 ELSE 0 END) as losses,
                ROUND(SUM(CASE WHEN exit_reason = 'TP' THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1) as win_rate,
                ROUND(SUM(net_pnl), 2) as total_pnl,
                ROUND(AVG(net_pnl), 2) as avg_pnl,
                ROUND(SUM(commission), 2) as total_fees,
                ROUND(AVG(entry_slippage_pts), 3) as avg_entry_slippage,
                ROUND(AVG(CASE WHEN exit_reason = 'SL' THEN stop_slippage_pts END), 3) as avg_stop_slippage,
                ROUND(AVG(duration_seconds), 0) as avg_duration_sec
            FROM trades
            WHERE exit_reason IS NOT NULL
        '''
        params = []
        if time_period:
            sql += ' AND time_period = ?'
            params.append(time_period)
        if risk_range:
            sql += ' AND risk_range = ?'
            params.append(risk_range)
        if setup:
            sql += ' AND setup = ?'
            params.append(setup)
        if days:
            sql += ' AND trade_date >= date("now", ?)'
            params.append(f'-{days} days')
        sql += ' GROUP BY time_period, risk_range, setup, n_value ORDER BY total_pnl DESC'
        return self.query(sql, params)

    def get_daily_summary(self, days=30):
        """Get daily stats for the last N days."""
        return self.query(
            'SELECT * FROM daily_stats ORDER BY trade_date DESC LIMIT ?',
            [days]
        )

    def get_period_pnl(self):
        """Current-week and current-month PNL from closed trades."""
        week = self.query(
            "SELECT ROUND(SUM(net_pnl), 2) as pnl, COUNT(*) as trades "
            "FROM trades WHERE exit_reason IS NOT NULL "
            "AND strftime('%Y-%W', trade_date) = strftime('%Y-%W', 'now')"
        )
        month = self.query(
            "SELECT ROUND(SUM(net_pnl), 2) as pnl, COUNT(*) as trades "
            "FROM trades WHERE exit_reason IS NOT NULL "
            "AND strftime('%Y-%m', trade_date) = strftime('%Y-%m', 'now')"
        )
        return {
            'week_pnl': (week[0]['pnl'] or 0) if week else 0,
            'week_trades': (week[0]['trades'] or 0) if week else 0,
            'month_pnl': (month[0]['pnl'] or 0) if month else 0,
            'month_trades': (month[0]['trades'] or 0) if month else 0,
        }

    def get_overall_stats(self, days=None):
        """Get aggregate stats across all trades."""
        sql = '''
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN exit_reason = 'TP' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN exit_reason = 'SL' THEN 1 ELSE 0 END) as losses,
                ROUND(SUM(CASE WHEN exit_reason = 'TP' THEN 1.0 ELSE 0 END) / NULLIF(COUNT(*), 0) * 100, 1) as win_rate,
                ROUND(SUM(net_pnl), 2) as total_pnl,
                ROUND(SUM(CASE WHEN net_pnl > 0 THEN net_pnl ELSE 0 END), 2) as gross_profit,
                ROUND(SUM(CASE WHEN net_pnl < 0 THEN net_pnl ELSE 0 END), 2) as gross_loss,
                ROUND(SUM(commission), 2) as total_fees,
                ROUND(AVG(net_pnl), 2) as avg_pnl_per_trade,
                ROUND(AVG(entry_slippage_pts), 3) as avg_entry_slippage,
                ROUND(AVG(CASE WHEN exit_reason = 'SL' THEN stop_slippage_pts END), 3) as avg_stop_slippage,
                SUM(contracts) as total_contracts,
                COUNT(DISTINCT trade_date) as trading_days
            FROM trades
            WHERE exit_reason IS NOT NULL
        '''
        params = []
        if days:
            sql += ' AND trade_date >= date("now", ?)'
            params.append(f'-{days} days')
        result = self.query(sql, params)
        if result:
            r = result[0]
            if r['gross_loss'] and r['gross_loss'] != 0:
                r['profit_factor'] = round(abs(r['gross_profit'] / r['gross_loss']), 2)
            else:
                r['profit_factor'] = None
            return r
        return {}

    def get_funnel(self, date=None):
        """FVG-to-trade conversion funnel."""
        where = 'WHERE trade_date = ?' if date else ''
        params = [date] if date else []

        fvgs = self.query(
            f'SELECT COUNT(*) as n FROM fvg_log {where}', params
        )[0]['n']
        mitigated = self.query(
            f'SELECT COUNT(*) as n FROM fvg_log {where} {"AND" if date else "WHERE"} mitigated = 1',
            params
        )[0]['n']
        placed = self.query(
            f'SELECT COUNT(*) as n FROM fvg_log {where} {"AND" if date else "WHERE"} trade_placed = 1',
            params
        )[0]['n']
        filled = self.query(
            f'SELECT COUNT(*) as n FROM trades {where} {"AND" if date else "WHERE"} actual_entry_price IS NOT NULL',
            params
        )[0]['n']
        won = self.query(
            f'SELECT COUNT(*) as n FROM trades {where} {"AND" if date else "WHERE"} exit_reason = "TP"',
            params
        )[0]['n']

        return {
            'fvgs_detected': fvgs,
            'fvgs_mitigated': mitigated,
            'trades_placed': placed,
            'trades_filled': filled,
            'trades_won': won,
        }

    def get_slippage_report(self, days=30):
        """Slippage analysis: entry and stop fill quality."""
        return self.query('''
            SELECT
                setup,
                COUNT(*) as trades,
                ROUND(AVG(entry_slippage_pts), 3) as avg_entry_slip,
                ROUND(MAX(entry_slippage_pts), 3) as max_entry_slip,
                ROUND(AVG(CASE WHEN exit_reason = 'SL' THEN stop_slippage_pts END), 3) as avg_stop_slip,
                ROUND(MAX(CASE WHEN exit_reason = 'SL' THEN stop_slippage_pts END), 3) as max_stop_slip,
                ROUND(SUM(commission), 2) as total_fees,
                ROUND(AVG(commission / NULLIF(contracts, 0)), 2) as avg_fee_per_contract
            FROM trades
            WHERE exit_reason IS NOT NULL
              AND trade_date >= date("now", ?)
            GROUP BY setup
        ''', [f'-{days} days'])

    def get_hourly_performance(self, days=None):
        """P&L by time-of-day (time_period)."""
        sql = '''
            SELECT
                time_period,
                COUNT(*) as trades,
                ROUND(SUM(net_pnl), 2) as total_pnl,
                ROUND(AVG(net_pnl), 2) as avg_pnl,
                SUM(CASE WHEN exit_reason = 'TP' THEN 1 ELSE 0 END) as wins,
                ROUND(SUM(CASE WHEN exit_reason = 'TP' THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1) as win_rate
            FROM trades WHERE exit_reason IS NOT NULL
        '''
        params = []
        if days:
            sql += ' AND trade_date >= date("now", ?)'
            params.append(f'-{days} days')
        sql += ' GROUP BY time_period ORDER BY time_period'
        return self.query(sql, params)

    # ── Alert Queue ─────────────────────────────────────────────────────

    def queue_alert(self, event_type, message):
        """Queue an alert for delivery with retry support."""
        conn = self._conn()
        conn.execute(
            'INSERT INTO alerts (event_type, message) VALUES (?, ?)',
            [event_type, message]
        )
        conn.commit()
        conn.close()

    def get_unsent_alerts(self, max_attempts=3, limit=10):
        """Get alerts that haven't been delivered yet."""
        return self.query(
            'SELECT * FROM alerts WHERE sent = 0 AND send_attempts < ? '
            'ORDER BY created_at ASC LIMIT ?',
            [max_attempts, limit]
        )

    def mark_alert_sent(self, alert_id):
        """Mark an alert as successfully sent."""
        conn = self._conn()
        conn.execute(
            'UPDATE alerts SET sent = 1, sent_at = CURRENT_TIMESTAMP WHERE id = ?',
            [alert_id]
        )
        conn.commit()
        conn.close()

    def mark_alert_failed(self, alert_id, error):
        """Increment attempt count and record error for a failed alert."""
        conn = self._conn()
        conn.execute(
            'UPDATE alerts SET send_attempts = send_attempts + 1, last_error = ? WHERE id = ?',
            [error, alert_id]
        )
        conn.commit()
        conn.close()

    # ── Queries ───────────────────────────────────────────────────────────

    def get_equity_curve(self, limit=500):
        """Trade-by-trade equity curve."""
        return self.query('''
            SELECT group_id, trade_date, entry_time, exit_time,
                   setup, time_period, exit_reason,
                   net_pnl, balance_after
            FROM trades
            WHERE exit_reason IS NOT NULL
            ORDER BY exit_time ASC
            LIMIT ?
        ''', [limit])
