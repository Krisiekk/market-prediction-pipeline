import unittest
import pandas as pd
from src.validation.calendar_audit import calendar_stats, target_audit


class CalendarAuditTests(unittest.TestCase):
    def test_h5_counts_weekend_rows_not_weekdays(self):
        data = pd.DataFrame({'datetime': pd.date_range('2026-09-18', periods=6),
                             'close': [10, 11, 12, 13, 14, 15]})
        audit = target_audit(data)
        row = audit.iloc[0]
        self.assertEqual(row.future_date, pd.Timestamp('2026-09-23'))
        self.assertEqual(row.weekday_records, 3)
        self.assertEqual(row.weekend_records, 2)
        self.assertEqual(row.calendar_days, 5)
        self.assertEqual(len(audit), 1)

    def test_calendar_counts_partition_rows_and_gaps(self):
        data = pd.DataFrame({'datetime': pd.to_datetime(['2026-09-18', '2026-09-19', '2026-09-21', '2026-09-25'])})
        stats = calendar_stats(data)
        self.assertEqual(sum(stats['weekday_counts'].values()), 4)
        self.assertEqual(stats['gaps_days'], {'1': 1, '2': 1, '3': 0, 'other': 1})


if __name__ == '__main__':
    unittest.main()
