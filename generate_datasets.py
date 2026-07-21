import pandas as pd
import numpy as np

np.random.seed(42)

def generate_samples(n, system):
    samples = []

    for _ in range(n):
        hour = np.random.randint(0, 24)

        scenario = np.random.choice(
            ['normal', 'peak', 'off_peak', 'crisis', 'degraded'],
            p=[0.45, 0.25, 0.10, 0.15, 0.05]
        )

        if scenario == 'normal':
            tps = np.random.uniform(300, 800)
            cpu = np.random.uniform(30, 60)
            lag = np.random.uniform(1, 30)
            errors = np.random.uniform(0, 0.3)
            deadlocks = np.random.randint(0, 3)
            memory = np.random.uniform(40, 65)
            connections = np.random.randint(20, 80)

        elif scenario == 'peak':
            tps = np.random.uniform(1000, 2000)
            cpu = np.random.uniform(75, 95)
            lag = np.random.uniform(20, 80)
            errors = np.random.uniform(0.1, 0.8)
            deadlocks = np.random.randint(2, 10)
            memory = np.random.uniform(70, 90)
            connections = np.random.randint(100, 200)

        elif scenario == 'off_peak':
            tps = np.random.uniform(50, 200)
            cpu = np.random.uniform(10, 30)
            lag = np.random.uniform(0, 10)
            errors = np.random.uniform(0, 0.1)
            deadlocks = np.random.randint(0, 1)
            memory = np.random.uniform(20, 45)
            connections = np.random.randint(1, 20)
            hour = np.random.choice([22, 23, 0, 1, 2, 3, 4, 5])

        elif scenario == 'crisis':
            tps = np.random.uniform(400, 900)
            cpu = np.random.uniform(60, 85)
            lag = np.random.uniform(90, 200)
            errors = np.random.uniform(0.8, 3.0)
            deadlocks = np.random.randint(4, 20)
            memory = np.random.uniform(75, 95)
            connections = np.random.randint(80, 180)

        else:  # degraded
            tps = np.random.uniform(100, 400)
            cpu = np.random.uniform(50, 80)
            lag = np.random.uniform(40, 120)
            errors = np.random.uniform(0.5, 2.0)
            deadlocks = np.random.randint(2, 8)
            memory = np.random.uniform(60, 85)
            connections = np.random.randint(40, 120)

        if system == 'postgresql':
            wal_buffers = np.random.uniform(0, 50)
            rollbacks = np.random.uniform(0, 80)
        elif system == 'mysql':
            wal_buffers = np.random.uniform(0, 30)
            rollbacks = np.random.uniform(0, 60)
        else:
            wal_buffers = np.random.uniform(0, 20)
            rollbacks = np.random.uniform(0, 40)

        # Labels 4-modes — règles IAHA strictes
        if lag > 100 or errors > 1.0 or deadlocks > 5:
            mode = 'CONSISTENCY'
        elif hour in [22, 23, 0, 1, 2, 3, 4, 5] and tps < 200:
            mode = 'ECONOMIC'
        elif tps > 1200 or cpu > 85:
            mode = 'CONSISTENCY'
        elif scenario == 'degraded' and lag > 40:
            mode = 'AVAILABILITY'
        elif tps < 500 and cpu < 50 and lag < 20:
            mode = 'PERFORMANCE'
        else:
            mode = 'CONSISTENCY'

        samples.append({
            'replication_lag_mb': round(lag, 2),
            'transactions_per_sec': round(tps, 2),
            'cpu_percent': round(cpu, 2),
            'latency_ms': round(np.random.uniform(5, 150), 2),
            'trans_rolled_back': round(rollbacks, 2),
            'deadlocks': deadlocks,
            'hour_of_day': hour,
            'memory_usage_pct': round(memory, 2),
            'connections_active': connections,
            'wal_buffers_full': round(wal_buffers, 2),
            'system': system,
            'mode': mode
        })

    return pd.DataFrame(samples)

print("Génération des datasets...")

pg_df = generate_samples(644, 'postgresql')
mysql_df = generate_samples(200, 'mysql')
mongo_df = generate_samples(200, 'mongodb')

pg_df.to_csv('dataset_postgresql_644.csv', index=False)
mysql_df.to_csv('dataset_mysql_200.csv', index=False)
mongo_df.to_csv('dataset_mongodb_200.csv', index=False)

for name, df in [('PostgreSQL', pg_df), ('MySQL', mysql_df), ('MongoDB', mongo_df)]:
    print(f"\n{name} (n={len(df)})")
    print(df['mode'].value_counts())
    pct = df['mode'].value_counts(normalize=True) * 100
    for mode, p in pct.items():
        print(f"  {mode}: {p:.1f}%")

print("\nDatasets générés avec succès.")