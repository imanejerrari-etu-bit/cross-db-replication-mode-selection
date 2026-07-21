import subprocess
import time
import csv
import statistics
from datetime import datetime

CONTEXT = "kind-iaha-mongodb"
NAMESPACE = "default"
RESULTS_FILE = "rto_mongodb_30tests.csv"

def mongo_eval(pod, script):
    cmd = [
        "kubectl", "exec", "-n", NAMESPACE, pod,
        f"--context={CONTEXT}", "--",
        "mongosh", "--quiet", "--eval", script
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return result.stdout.strip()

def run_kubectl(args):
    cmd = ["kubectl"] + args + [f"--context={CONTEXT}"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return result.stdout.strip()

def get_primary():
    for pod in ["mongodb-0", "mongodb-1", "mongodb-2"]:
        try:
            out = mongo_eval(pod,
                "rs.status().members.filter(m => m.stateStr === 'PRIMARY')"
                ".map(m => m.name)[0]")
            if out and "mongodb-" in out:
                host = out.split(".")[0].replace("'", "").replace('"', '').strip()
                return host
        except:
            continue
    return None

def get_online_count():
    for pod in ["mongodb-0", "mongodb-1", "mongodb-2"]:
        try:
            out = mongo_eval(pod,
                "rs.status().members.filter(m => "
                "m.stateStr === 'PRIMARY' || m.stateStr === 'SECONDARY').length")
            out = out.strip()
            if out.isdigit():
                return int(out)
        except:
            continue
    return 0

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
    print(f"  Membres actifs : {count}")

    primary = get_primary()
    if not primary:
        print("  Primary introuvable, skip")
        return None

    print(f"  Primary : {primary}")
    time.sleep(10)

    t_start = time.time()
    run_kubectl(["delete", "pod", primary, "-n", NAMESPACE])
    print(f"  Failover à {datetime.now().strftime('%H:%M:%S')}")

    time.sleep(15)

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
    print(f"  RTO (3 actifs) : {rto:.1f}s")

    print("  Stabilisation 60s...")
    time.sleep(60)

    return {
        "test": test_num,
        "former_primary": primary,
        "new_primary": new_primary if new_primary else "unknown",
        "promotion_time_s": round(t_promotion, 2),
        "rto_s": round(rto, 2),
        "timestamp": datetime.now().isoformat()
    }

def main():
    print("=== IAHA-X MongoDB Failover Tests — 30 runs ===")
    print(f"Cluster  : {CONTEXT}")
    print(f"Résultats: {RESULTS_FILE}")
    print(f"Durée estimée : ~60 minutes")

    print(f"\nÉtat initial :")
    print(mongo_eval("mongodb-0",
        "rs.status().members.map(m => m.name + ' : ' + m.stateStr).join('\\n')"))

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
                if result and result["rto_s"] > 0:
                    results.append(result)
                    writer.writerow(result)
                    f.flush()
                    print(f"  ✓ Test {i} enregistré — RTO: {result['rto_s']}s")
            except Exception as e:
                print(f"  ✗ Test {i} échoué : {e}")

    rtos = [r["rto_s"] for r in results]
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