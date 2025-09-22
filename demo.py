import asyncio
import random
import uuid
import os
import time
from collections import deque
from typing import Dict, Set, List

# --- 核心数据结构 (无变化) ---

class Sample:
    """样本的数据结构，与之前版本保持一致。"""
    def __init__(self, query_id: str, query_data: str):
        self.id: str = str(uuid.uuid4())
        self.query_id: str = query_id
        self.state: str = "unprocess"
        self.history: list = [query_data]
        self.rollout_result: str = None
        self.reward: float = None

    def __repr__(self):
        return f"Sample(id={self.id[-6:]}, state='{self.state}')"

# --- 中央调度器 (逻辑修改) ---

class SampleManager:
    """
    中央数据调度器。
    """
    def __init__(self):
        self._buffer: Dict[str, Sample] = {}
        self._state_index: Dict[str, Set[str]] = {
            "unprocess": set(),
            "in_rollout": set(),
            "generate_done": set(),
            "in_reward": set(),
            "processed": set(),
            "in_training": set(),
            "done": set(), # 新增最终状态
        }
        self._lock = asyncio.Lock()

    async def add_samples(self, samples: list):
        async with self._lock:
            for sample in samples:
                if sample.id not in self._buffer:
                    self._buffer[sample.id] = sample
                    self._state_index[sample.state].add(sample.id)

    async def request_samples(self, from_state: str, to_state: str, count: int) -> list:
        async with self._lock:
            ids_to_provide = list(self._state_index[from_state])[:count]
            if not ids_to_provide: return []

            samples_to_provide = []
            for sample_id in ids_to_provide:
                self._state_index[from_state].remove(sample_id)
                self._state_index[to_state].add(sample_id)
                sample = self._buffer[sample_id]
                sample.state = to_state
                samples_to_provide.append(sample)
            return samples_to_provide

    # --- 方法已修改：从单个更新变为批量更新 ---
    async def update_samples_state(self, samples: List[Sample], from_state: str, to_state: str):
        """接收 worker 处理完成的一批样本，并更新其状态。"""
        async with self._lock:
            for sample in samples:
                if sample.id in self._buffer: # 确保样本还存在
                    # 从旧状态索引中移除
                    self._state_index[from_state].discard(sample.id)
                    # 添加到新状态索引
                    self._state_index[to_state].add(sample.id)
                    # 更新样本对象自身的状态
                    sample.state = to_state


    async def get_status_snapshot(self) -> Dict[str, int]:
        async with self._lock:
            return {state: len(ids) for state, ids in self._state_index.items()}

# --- 自治的 Worker Manager ---

class RolloutManager:
    def __init__(self, sample_manager: SampleManager, concurrency_level: int = 4):
        self.sample_manager = sample_manager; self.concurrency_level = concurrency_level; self.active_tasks: Set[asyncio.Task] = set()
    async def _process_one(self, sample: Sample):
        try:
            await asyncio.sleep(random.uniform(0.5, 3.0)); sample.rollout_result = f"Rollout for {sample.history[0]}"; sample.history.append(sample.rollout_result); 
            # 注意：现在单个更新也通过批量接口，保持一致性
            await self.sample_manager.update_samples_state([sample], "in_rollout", "generate_done")
        except asyncio.CancelledError: pass
    async def run(self):
        while True:
            if len(self.active_tasks) < self.concurrency_level:
                needed = self.concurrency_level - len(self.active_tasks); new_samples = await self.sample_manager.request_samples("unprocess", "in_rollout", needed)
                for sample in new_samples: task = asyncio.create_task(self._process_one(sample)); self.active_tasks.add(task); task.add_done_callback(self.active_tasks.discard)
            await asyncio.sleep(0.1)

class RewardManager:
    def __init__(self, sample_manager: SampleManager, concurrency_level: int = 8):
        self.sample_manager = sample_manager; self.concurrency_level = concurrency_level; self.active_tasks: Set[asyncio.Task] = set()
    async def _process_one(self, sample: Sample):
        try:
            await asyncio.sleep(random.uniform(0.1, 0.5)); sample.reward = random.choice([0.0, 1.0]); 
            await self.sample_manager.update_samples_state([sample], "in_reward", "processed")
        except asyncio.CancelledError: pass
    async def run(self):
        while True:
            if len(self.active_tasks) < self.concurrency_level:
                needed = self.concurrency_level - len(self.active_tasks); new_samples = await self.sample_manager.request_samples("generate_done", "in_reward", needed)
                for sample in new_samples: task = asyncio.create_task(self._process_one(sample)); self.active_tasks.add(task); task.add_done_callback(self.active_tasks.discard)
            await asyncio.sleep(0.1)

# --- 训练工作单元 (逻辑修改) ---
class TrainingManager:
    def __init__(self, sample_manager: SampleManager, batch_size: int = 32, total_steps: int = 10):
        self.sample_manager = sample_manager
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.model_version = 0
        self.steps_done = 0
        self.finished_event = asyncio.Event()

    async def _train_step(self, batch: List[Sample]):
        """模拟一个训练步骤。"""
        print(f"🏋️ [TrainingManager] Starting training step {self.steps_done + 1}/{self.total_steps} with a batch of {len(batch)}.")
        await asyncio.sleep(5.0)
        self.model_version += 1
        self.steps_done += 1
        print(f"✅ [TrainingManager] Training complete. Model updated to version {self.model_version}.")
        
        # --- 核心改动 ---
        # 不再移除样本，而是将其状态更新为 "done"
        await self.sample_manager.update_samples_state(batch, "in_training", "done")
        print(f"➡️  [TrainingManager] {len(batch)} samples marked as 'done'.")


    async def run(self):
        """主运行循环，拉取数据并执行训练。"""
        while self.steps_done < self.total_steps:
            training_batch = await self.sample_manager.request_samples("processed", "in_training", self.batch_size)
            
            if len(training_batch) == self.batch_size:
                await self._train_step(training_batch)
            else:
                # 如果数据不够，则退还已取出的样本
                if training_batch:
                    await self.sample_manager.update_samples_state(training_batch, "in_training", "processed")
                await asyncio.sleep(0.5)
        
        print("🎉 [TrainingManager] All training steps completed!")
        self.finished_event.set()


# --- 数据源模拟 ---
async def data_source(sample_manager: SampleManager, count: int):
    for i in range(count):
        sample = Sample(f"query_{i}", f"Initial prompt {i}")
        await sample_manager.add_samples([sample])

# --- 实时监控组件 (无变化) ---
async def status_monitor_worker(sample_manager: SampleManager, training_manager: TrainingManager, start_time: float):
    """监控任务，现在会自动显示 "done" 状态。"""
    while not training_manager.finished_event.is_set():
        snapshot = await sample_manager.get_status_snapshot()
        os.system('cls' if os.name == 'nt' else 'clear')
        elapsed_time = time.time() - start_time
        
        print("--- Custom RL Framework - Live Status ---")
        print(f"Time Elapsed: {elapsed_time:.1f}s")
        print(f"Model Version: {training_manager.model_version} | Training Steps: {training_manager.steps_done}/{training_manager.total_steps}")
        print("-" * 40)
        
        total_in_buffer = sum(snapshot.values())
        for state, count in sorted(snapshot.items(), key=lambda item: list(sample_manager._state_index.keys()).index(item[0])):
            if total_in_buffer > 0:
                bar_length = 30
                filled_length = int(bar_length * count / total_in_buffer) if total_in_buffer > 0 else 0
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                print(f"{state:<15} | {bar} | {count}")
            else:
                print(f"{state:<15} | {'-' * 30} | {count}")
        print("-" * 40)
        await asyncio.sleep(0.5)
    print("Monitor exiting as training is complete.")

# --- 主程序入口 (无变化) ---
async def main():
    BATCH_SIZE = 32
    TOTAL_STEPS = 10
    TOTAL_SAMPLES = BATCH_SIZE * TOTAL_STEPS + 50 

    start_time = time.time()
    
    sample_manager = SampleManager()
    rollout_manager = RolloutManager(sample_manager, concurrency_level=16)
    reward_manager = RewardManager(sample_manager, concurrency_level=16)
    training_manager = TrainingManager(sample_manager, batch_size=BATCH_SIZE, total_steps=TOTAL_STEPS)
    
    rollout_task = asyncio.create_task(rollout_manager.run())
    reward_task = asyncio.create_task(reward_manager.run())
    training_task = asyncio.create_task(training_manager.run())
    monitor_task = asyncio.create_task(status_monitor_worker(sample_manager, training_manager, start_time))

    asyncio.create_task(data_source(sample_manager, TOTAL_SAMPLES))

    await training_manager.finished_event.wait()

    print("\n--- Training finished. Stopping all services... ---")
    monitor_task.cancel(); rollout_task.cancel(); reward_task.cancel(); training_task.cancel()
    await asyncio.gather(monitor_task, rollout_task, reward_task, training_task, return_exceptions=True)

    print("--- Demo Finished ---")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting.")