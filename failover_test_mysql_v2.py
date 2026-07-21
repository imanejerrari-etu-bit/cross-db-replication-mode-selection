import subprocess
import time
import csv
import statistics
from datetime import datetime

CONTEXT = "kind-iaha-mysql"
NAMESPACE = "default"
RESULTS_FILE = "rto_mysql_30tests.csv"
MYSQL_PASSWORD = "IAHApass2024!"

def mysql_query(pod, query):
    cmd = [
        "kubectl", "exec", "-n", NAMESPACE, pod,
        f"--context={CONTEXT}", "-c", "mysql", "--",
        "mysql", "-uroot", f"-p{MYSQL_PASSWORD}", "-sN", "-e", query
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return result.stdout.strip()

def run_kubectl(args):
    cmd = ["kubectl"] + args + [f"--context={CONTEXT}"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return result.stdout.strip()

def get_online_members():
    for pod in ["iaha-mysql-cluster-0", "iaha-mysql-cluster-1", "iaha-mysql-cluster-2"]:
        try:
            out = mysql_query(pod,
                "SELECT MEMBER_HOST, MEMBER_ROLE, MEMBER_STATE "
                "FROM performance_schema.replication_group_members;")
            if out:
                return out
        except:
            continue
    return ""

def get_primary():
    for pod in ["iaha-mysql-cluster-0", "iaha-mysql-cluster-1", "iaha-mysql-cluster-2"]:
        try:
            out = mysql_query(pod,
                "SELECT MEMBER_HOST FROM performance_schema.replication_group_members "
                "WHERE MEMBER_ROLE='PRIMARY' AND MEMBER_STATE='ONLINE';")
            for line in out.split("\n"):
                if "iaha-mysql-cluster" in line:
                    return line.strip().split(".")[0]
        except:
            continue
    return None

def get_online_count():
    for pod in ["iaha-mysql-cluster-0", "iaha-mysql-cluster-1", "iaha-mysql-cluster-2"]:
        try:
            out = mysql_query(pod,
                "SELECT COUNT(*) FROM performance_schema.replication_group_members "
                "WHERE MEMBER_STATE='ONLINE';")
            for line in out.split("\n"):
                if line.strip().isdigit():
                    return int(line.strip())
        except:
            continue
    return 0

def restart_gr():
    print("  → Redémarrage GR...")
    for pod in ["iaha-mysql-cluster-0", "iaha-mysql-cluster-1", "iaha-mysql-cluster-2"]:
        try:
            mysql_query(pod, "STOP GROUP_REPLICATION;")
        except:
            pass
    time.sleep(5)
    try:
        mysql_query("iaha-mysql-cluster-0",
            "SET GLOBAL group_replication_bootstrap_group=ON; "
            "START GROUP_REPLICATION; "
            "SET GLOBAL group_replication_bootstrap_group=OFF;")
        time.sleep(10)
        for pod in ["iaha-mysql-cluster-1", "iaha-mysql-cluster-2"]:
            try:
                mysql_query(pod, "START GROUP_REPLICATION;")
                time.sleep(5)
            except:
                pass
    except Exception as e:
        print(f"  → Erreur restart: {e}")

def wait_for_healthy(timeout=180):
    start = time.time()
    while time.time() - start < timeout:
        try:
            count = get_online_count()
            if count >= 3:
                return time.time() - start
        except:
            pass
        time.sleep(3)
    return -1

def run_failover_test(test_num):
    print(f"\n--- Test {test_num}/30 ---")

    count = get_online_count()
    if count < 2:
        print(f"  Cluster dégradé ({count} ONLINE), redémarrage...")
        restart_gr()
        time.sleep(20)

    primary = get_primary()
    if not primary:
        print("  Primary introuvable après recovery, skip")
        return None

    print(f"  Primary : {primary}")
    print(f"  Membres ONLINE : {count}")

    time.sleep(15)

    t_start = time.time()
    run_kubectl(["delete", "pod", primary, "-n", NAMESPACE])
    print(f"  Failover à {datetime.now().strftime('%H:%M:%S')}")

    time.sleep(20)

    new_primary = None
    for _ in range(60):
        candidate = get_primary()
        if candidate and candidate != primary:
            new_primary = candidate
            break
        time.sleep(3)

    t_promotion = time.time() - t_start
    print(f"  Nouveau primary : {new_primary} en {t_promotion:.1f}s")

    rto = wait_for_healthy()
    print(f"  RTO (3 ONLINE) : {rto:.1f}s")

    print("  Stabilisation 120s...")
    time.sleep(120)

    return {
        "test": test_num,
        "former_primary": primary,
        "new_primary": new_primary if new_primary else "unknown",
        "promotion_time_s": round(t_promotion, 2),
        "rto_s": round(rto, 2),
        "timestamp": datetime.now().isoformat()
    }

def main():
    print("=== IAHA-X MySQL GR Failover Tests — 30 runs ===")
    print(f"Cluster  : {CONTEXT}")
    print(f"Résultats: {RESULTS_FILE}")
    print(f"Durée estimée : ~105 minutes")

    print(f"\nÉtat initial :")
    print(get_online_members())

    results = []

    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "test", "former_primary", "new_primary",
            "promotion_time_s", "rto_s", "timestamp"
        ])
        writer.writeheader()

        for i in range(1, 31):
            try:
                result = run_failover_test(i)
                if result:
                    results.append(result)
                    writer.writerow(result)
                    f.flush()
                    print(f"  ✓ Test {i} enregistré — RTO: {result['rto_s']}s")
            except Exception as e:
                print(f"  ✗ Test {i} échoué : {e}")

    rtos = [r["rto_s"] for r in results if r["rto_s"] > 0]
    if rtos:
        print(f"\n=== Résultats finaux ({len(rtos)} tests) ===")
        print(f"RTO moyen  : {statistics.mean(rtos):.2f}s")
        print(f"RTO médian : {statistics.median(rtos):.2f}s")
        print(f"RTO min    : {min(rtos):.2f}s")
        print(f"RTO max    : {max(rtos):.2f}s")
        if len(rtos) > 1:
            print(f"Écart-type : {statistics.stdev(rtos):.2f}s")

if __name__ == "__main__":
    main()