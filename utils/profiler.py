from dataclasses import dataclass, field
import time


@dataclass
class Profiler:
    timings: dict = field(default_factory=dict)
    values: dict = field(default_factory=dict)
    _starts: dict = field(default_factory=dict)

    def start(self, name: str):
        self._starts[name] = time.perf_counter()

    def stop(self, name: str):
        if name not in self._starts:
            return
        elapsed = (time.perf_counter() - self._starts.pop(name)) * 1000
        self.timings[name] = elapsed

    def set(self, key: str, value):
        self.values[key] = value

    def report(self):
        print("\n" + "=" * 60)
        print("HERMES PROFILER")
        print("=" * 60)

        if self.values:
            print("\nVALUES")
            print("-" * 60)
            for k, v in self.values.items():
                print(f"{k:25}: {v}")

        if self.timings:
            print("\nTIMINGS")
            print("-" * 60)
            total = 0

            for k, v in self.timings.items():
                if k == "TOTAL":
                    continue
                print(f"{k:25}: {v:8.2f} ms")
                total += v

            print("-" * 60)
            print(f"{'Measured Total':25}: {total:8.2f} ms")

        print("=" * 60)
