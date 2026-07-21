import subprocess
import time
import csv
import statistics
from datetime import datetime

CONTEXT = "kind-iaha-mysql"
NAMESPACE = "default"
RESULTS_FILE = "rto_mysql_30tests.csv"
MYSQL_PASSWORD = "IAHApass2024!"
EXEC_POD = "iaha-mysql-cluster-0"

def mysql_query(query):
    cmd = [
        "kubectl", "exec", "-n", NAMESPACE, EXEC_POD,
        f"--context={CONTEXT}", "-c", "mysql", "--",
        "mysql", "-uroot", f"-p{MYSQL_PASSWORD}", "-sN", "-e", query
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()

def run_kubectl(args):
    cmd = ["kubectl"] + args + [f"--context={CONTEXT}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()

def get_primary():
    out = mysql_query(
        "SELECT MEMBER_HOST FROM performance_schema.replication_group_members "
        "WHERE MEMBER_ROLE='PRIMARY';"
    )
    for line in out.split("\n"):
        line = line.strip()
        if "iaha-mysql-cluster" in line:
            return line.split(".")[0]
    return None

def get_cluster_online():
    out = mysql_query(
        "SELECT COUNT(*) FROM performance_schema.replication_group_members "
        "WHERE MEMBER_STATE='ONLINE';"
    )
    for line in out.split("\n"):
        line = line.strip()
        if line.isdigit():
            return int(line)
    return 0

def wait_for_healthy(timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        try:
            online = get_cluster_online()
            if online >= 3:
                return time.time() - start
        except:
            pass
        time.sleep(2)
    return -1

def run_failover_test(test_num):
    print(f"\n--- Test {test_num}/30 ---")

    primary = get_primary()
    if not primary:
        print("Primary introuvable, skip")
        return None
    print(f"Primary avant failover : {primary}")

    # Stabilisation avant failover
    time.sleep(10)

    t_start = time.time()
    run_kubectl(["delete", "pod", primary, "-n", NAMESPACE])
    print(f"Pod {primary} supprimé à {datetime.now().strftime('%H:%M:%S')}")

    # Attendre que le pod soit vraiment parti
    print("Attente disparition pod...")
    time.sleep(15)

    # Attendre nouveau primary différent
    new_primary = None
    attempts = 0
    while attempts < 90:
        try:
            candidate = get_primary()
            if candidate and candidate != primary:
                new_primary = candidate
                break
        except:
            pass
        time.sleep(2)
        attempts += 1

    t_promotion = time.time() - t_start
    print(f"Nouveau primary : {new_primary} en {t_promotion:.2f}s")

    # Attendre 3 membres ONLINE
    rto = wait_for_healthy(timeout=180)
    print(f"Cluster healthy (3 ONLINE) en {rto:.2f}s")

    # Longue stabilisation entre tests
    print("Stabilisation 90s...")
    time.sleep(90)

    return {
        "test": test_num,
        "former_primary": primary,
        "new_primary": new_primary,
        "promotion_time_s": round(t_promotion, 2),
        "rto_s": round(rto, 2),
        "timestamp": datetime.now().isoformat()
    }
def main():
    print("=== IAHA-X MySQL Failover Test Suite ===")
    print(f"Cluster : {CONTEXT}")
    print(f"Résultats : {RESULTS_FILE}")

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
                    print(f"✓ Test {i} enregistré — RTO: {result['rto_s']}s")
            except Exception as e:
                print(f"✗ Test {i} échoué : {e}")

    rtos = [r["rto_s"] for r in results if r["rto_s"] > 0]
    if rtos:
        print(f"\n=== Résultats finaux ({len(rtos)} tests) ===")
        print(f"RTO moyen  : {statistics.mean(rtos):.2f}s")
        print(f"RTO médian : {statistics.median(rtos):.2f}s")
        print(f"RTO min    : {min(rtos):.2f}s")
        print(f"RTO max    : {max(rtos):.2f}s")
        print(f"Écart-type : {statistics.stdev(rtos):.2f}s")

if __name__ == "__main__":
    main()