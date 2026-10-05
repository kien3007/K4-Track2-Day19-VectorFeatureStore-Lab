"""Demo script for HybridMemoryAgent — Personal AI Memory System.

Executes 5 benchmark queries across episodic memory and Feast profile features.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bonus.agent import HybridMemoryAgent


def main() -> int:
    print("=" * 70)
    print("  DAY 19 BONUS CHALLENGE: PERSONAL AI HYBRID MEMORY DEMO")
    print("=" * 70)

    # 1. Initialize Agent
    agent = HybridMemoryAgent(
        feature_store_path=ROOT / "app" / "feast_repo"
    )

    # 2. Seed realistic Episodic Memories for user u_001
    sample_notes = [
        "Kubernetes kiến trúc: Đã đọc bài viết về K8s Pods, ReplicaSet và Deployment trên cụm bare-metal.",
        "Thiết lập Auto-scaling: Cấu hình Horizontal Pod Autoscaler (HPA) dựa trên CPU utilization và custom metrics Prometheus.",
        "Bảo mật Cloud & K8s: Nghiên cứu NetworkPolicy, RBAC, và quản lý bí mật (Secrets management) bằng HashiCorp Vault.",
        "Tối ưu chi phí hạ tầng: Kế hoạch cắt giảm 35% chi phí điện toán bằng AWS Spot Instances và Karpenter node autoscaler.",
        "Frontend State Management: Đọc tài liệu so sánh Zustand và Redux Toolkit cho dự án Next.js tại công ty.",
    ]

    print(f"\n[1] Seeding {len(sample_notes)} episodic memories for user 'u_001'...")
    for note in sample_notes:
        agent.remember(note, user_id="u_001")
    print("  -> Đã nạp thành công ký ức vào Qdrant Vector Store + BM25 Lexical Index.")

    # 3. Five Benchmark Queries from BONUS-CHALLENGE.md
    test_queries = [
        ("Query 1: Hỏi đơn giản (Direct keyword/vector hit)",
         "Tôi đã đọc gì về Kubernetes?"),

        ("Query 2: Hỏi cần profile context (Dựa vào topic_affinity & preferred_language)",
         "Recommend đọc gì tiếp theo?"),

        ("Query 3: Hỏi cần fresh activity (Dựa vào queries_last_hour & distinct_topics)",
         "Tôi đang quan tâm gì gần đây?"),

        ("Query 4: Hỏi diễn đạt lại / Paraphrase (Vector search chiếm ưu thế)",
         "Tài liệu về tự động mở rộng hạ tầng?"),

        ("Query 5: Hỏi hỗn hợp / Mixed (Hybrid Search RRF + User Profile)",
         "Cho tôi summary cloud security"),
    ]

    print("\n" + "=" * 70)
    print("  RUNNING 5 TEST QUERIES")
    print("=" * 70)

    for title, query in test_queries:
        print(f"\n>>> {title}")
        print(f"Câu hỏi: \"{query}\"")
        context = agent.recall(query, user_id="u_001", top_k=2)
        print(context)
        print("-" * 70)

    print("\n[SUCCESS] Hoàn thành 5/5 truy vấn thử nghiệm mẫu. Exit code 0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
