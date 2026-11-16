# Copyright (c) 2026. The SimGrid Team. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the license (GNU LGPL) which comes with this package.

"""
Dynamic list scheduling of a workflow on hosts that fail and come back.

The workflow is read from a WfFormat JSON file (https://github.com/wfcommons/wfformat), the format of the C++
dag-from-json examples. A controller actor places each ready task on an idle host, starts the transfers of its input
files and its execution, and reacts to completions, host failures and reboots. When a host is turned off, the
execution running on it fails and the task is placed again elsewhere. Output files stay on the disk of the host that
wrote them, so they can be fetched again once that host is back on.

Policies (--policy):
  minmin  Min-Min, as in examples/cpp/dag-scheduling: the (task, host) pair that finishes first goes first.
  heft    Tasks by decreasing upward rank, each on the host where it finishes first (a dynamic HEFT).
  risk    Like heft, but a host's finish time is penalised by the time expected to be lost if that host fails
          during the task: EFT + w * p_fail * (EFT - now), with p_fail = 1 - exp(-rate * (EFT - now)) and
          w = 1 + lambda * criticality / mean(criticality).

Task criticality (--criticality) for the risk policy: uniform (all tasks alike), rank (upward rank), descendants,
critical-path (1 on the critical path, else 0), centrality (0.35 PageRank + 0.30 betweenness + 0.20 degree
+ 0.15 closeness on the DAG, divided by its maximum), entropy (Shannon entropy of the shares of a task's output bytes
among its children) or dependency (0.40 centrality + 0.35 entropy + 0.25 critical-path, the centrality-entropy
index of P. Pokharel).

A chaos actor turns hosts off at seeded random dates drawn from --rate (failures per host per second; hosts w03,
w07, w11 and w15 fail four times as often) and back on --downtime seconds later, like the cmonkey plugin does for a
single host. The plugin itself can be used instead (--rate 0 --cfg=plugin:cmonkey ...).

Usage: dag-failure-scheduling.py workflow.json [--policy P] [--criticality C] [--rate R] [--seed S] [--json]
"""

import argparse
import json
import math
import random
import sys

from simgrid import (ActivitySet, Comm, Engine, Exec, HostFailureException, NetworkFailureException,
                     TimeoutException, sg_host_energy_plugin_init, sg_host_energy_update_all, this_actor)

SPEEDS = [0.5e9, 1e9, 1.5e9, 2e9]                         # flop/s of the four host classes
POWER = ["60:70:110", "80:95:160", "95:115:210", "110:135:260"]  # idle:epsilon:max (W) of each class
BW, LAT, BB_BW, BB_LAT = 1.25e8, 50e-6, 1.25e9, 10e-6     # private links and backbone
REF_SPEED = 1e9                                            # traced runtimes are scaled to flops with this speed


def load_workflow(path):
    """Tasks (in topological order), their flops, parents, children and the bytes on each edge."""
    with open(path) as f:
        wf = json.load(f)["workflow"]
    spec, execution = wf["specification"], wf["execution"]
    runtime = {t["id"]: t["runtimeInSeconds"] for t in execution["tasks"]}
    size = {f["id"]: f["sizeInBytes"] for f in spec["files"]}
    tasks = {t["id"]: t for t in spec["tasks"]}
    order, indeg = [], {t: len(tasks[t]["parents"]) for t in tasks}
    stack = sorted((t for t in tasks if indeg[t] == 0), reverse=True)
    while stack:
        t = stack.pop()
        order.append(t)
        for c in sorted(tasks[t]["children"], reverse=True):
            indeg[c] -= 1
            if indeg[c] == 0:
                stack.append(c)
    edge = {(p, c): sum(size[x] for x in set(tasks[p]["outputFiles"]) & set(tasks[c]["inputFiles"]))
            for p in tasks for c in tasks[p]["children"]}
    return {"order": order, "flops": {t: runtime[t] * REF_SPEED for t in tasks},
            "parents": {t: list(tasks[t]["parents"]) for t in tasks},
            "children": {t: list(tasks[t]["children"]) for t in tasks}, "edge": edge}


def transfer_time(nbytes):
    return 0.0 if nbytes <= 0 else 2 * LAT + BB_LAT + nbytes / BW


def ranks(wf):
    """Upward and downward ranks with mean computation costs, as in HEFT."""
    mean_inv = sum(1 / s for s in SPEEDS) / len(SPEEDS)
    cost = {t: wf["flops"][t] * mean_inv for t in wf["order"]}
    up, down = {}, {}
    for t in reversed(wf["order"]):
        up[t] = cost[t] + max((transfer_time(wf["edge"][t, c]) + up[c] for c in wf["children"][t]), default=0.0)
    for t in wf["order"]:
        down[t] = max((down[p] + cost[p] + transfer_time(wf["edge"][p, t]) for p in wf["parents"][t]), default=0.0)
    return up, down


def centrality(wf):
    """0.35 PageRank + 0.30 betweenness + 0.20 degree + 0.15 closeness on the directed DAG, divided by the max."""
    nodes, succ, pred = wf["order"], wf["children"], wf["parents"]
    n = len(nodes)
    pr = dict.fromkeys(nodes, 1.0 / n)                     # PageRank, alpha 0.85, uniform teleport and dangling
    for _ in range(100):
        dangling = sum(pr[v] for v in nodes if not succ[v])
        new = {v: 0.15 / n + 0.85 * dangling / n for v in nodes}
        for v in nodes:
            for c in succ[v]:
                new[c] += 0.85 * pr[v] / len(succ[v])
        done = sum(abs(new[v] - pr[v]) for v in nodes) < n * 1e-6
        pr = new
        if done:
            break
    bc = dict.fromkeys(nodes, 0.0)                         # Brandes betweenness, unweighted
    for s in nodes:
        stack, preds, sigma, dist, queue = [], {v: [] for v in nodes}, dict.fromkeys(nodes, 0), {s: 0}, [s]
        sigma[s] = 1
        while queue:
            v = queue.pop(0)
            stack.append(v)
            for w in succ[v]:
                if w not in dist:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    preds[w].append(v)
        delta = dict.fromkeys(nodes, 0.0)
        while stack:
            w = stack.pop()
            for v in preds[w]:
                delta[v] += sigma[v] / sigma[w] * (1 + delta[w])
            if w != s:
                bc[w] += delta[w]
    scale = 1.0 / ((n - 1) * (n - 2)) if n > 2 else 1.0
    closeness = {}
    for u in nodes:                                        # distances *to* u, as networkx does for digraphs
        dist, queue = {u: 0}, [u]
        while queue:
            v = queue.pop(0)
            for w in pred[v]:
                if w not in dist:
                    dist[w] = dist[v] + 1
                    queue.append(w)
        total, reach = sum(dist.values()), len(dist) - 1
        closeness[u] = reach / total * reach / (n - 1) if total > 0 and n > 1 else 0.0
    comp = {v: 0.35 * pr[v] + 0.30 * bc[v] * scale + 0.20 * (len(succ[v]) + len(pred[v])) / (n - 1)
            + 0.15 * closeness[v] for v in nodes}
    top = max(comp.values())
    return {v: comp[v] / top if top > 0 else 0.0 for v in nodes}


def flow_entropy(wf):
    """Normalised Shannon entropy of the shares of a task's output bytes among its children."""
    out = {}
    for t in wf["order"]:
        kids = wf["children"][t]
        if len(kids) < 2:
            out[t] = 0.0
            continue
        sizes = [wf["edge"][t, c] for c in kids]
        total = sum(sizes)
        shares = [s / total for s in sizes] if total > 0 else [1 / len(kids)] * len(kids)
        out[t] = -sum(p * math.log(p) for p in shares if p > 0) / math.log(len(kids))
    return out


def criticality(wf, kind):
    up, down = ranks(wf)
    length = max(up.values())
    on_path = {t: 1.0 if abs(up[t] + down[t] - length) <= 1e-9 * length else 0.0 for t in wf["order"]}
    if kind == "uniform":
        raw = dict.fromkeys(wf["order"], 1.0)
    elif kind == "rank":
        raw = up
    elif kind == "descendants":
        below = {}
        for t in reversed(wf["order"]):
            below[t] = set(wf["children"][t]).union(*(below[c] for c in wf["children"][t]))
        raw = {t: float(len(below[t])) for t in wf["order"]}
    elif kind == "critical-path":
        raw = on_path
    elif kind == "centrality":
        raw = centrality(wf)
    elif kind == "entropy":
        raw = flow_entropy(wf)
    elif kind == "dependency":
        cen, ent = centrality(wf), flow_entropy(wf)
        raw = {t: 0.40 * cen[t] + 0.35 * ent[t] + 0.25 * on_path[t] for t in wf["order"]}
    else:
        raise ValueError(f"Unknown criticality '{kind}'")
    mean = sum(raw.values()) / len(raw)
    return {t: raw[t] / mean if mean > 0 else 1.0 for t in raw}


def build_platform(e):
    """16 workers in four speed classes, plus a controller that never fails and runs no task."""
    zone = e.netzone_root
    backbone = zone.add_link("backbone", BB_BW).set_latency(BB_LAT).seal()
    hosts, links = [], []
    ctl = zone.add_host("ctl", REF_SPEED)
    ctl.set_property("wattage_per_state", "0:0:0")
    ctl.set_property("wattage_off", "0")
    ctl.seal()
    for i in range(16):
        h = zone.add_host(f"w{i:02d}", SPEEDS[i // 4])
        h.set_property("wattage_per_state", POWER[i // 4])
        h.set_property("wattage_off", "10")
        hosts.append(h.seal())
    for h in [ctl] + hosts:
        links.append(zone.add_link(f"l_{h.name}", BW).set_latency(LAT).seal())
    every = [ctl] + hosts
    for a in range(len(every)):
        for b in range(a + 1, len(every)):
            zone.add_route(every[a], every[b], [links[a], backbone, links[b]])
    zone.seal()
    return ctl, hosts


def failure_schedule(hosts, rate, downtime, seed, horizon):
    """Seeded Poisson failures: (date, host, back_on) events. Hosts w03, w07, w11 and w15 fail 4 times as often."""
    rates, events = {}, []
    for i, h in enumerate(hosts):
        rates[h.name] = rate * (4 if i % 4 == 3 else 1)
        rng, t = random.Random(seed * 1000 + i), 0.0
        while rates[h.name] > 0:
            t += rng.expovariate(rates[h.name])
            if t > horizon:
                break
            events += [(t, i, False), (t + downtime, i, True)]
            t += downtime
    return rates, [(date, hosts[i], on) for date, i, on in sorted(events, key=lambda ev: (ev[0], ev[1]))]


def chaos(events):
    """Turn hosts off and back on at the given dates, as the cmonkey plugin does for a single host."""
    for date, host, back_on in events:
        this_actor.sleep_until(date)
        if back_on:
            host.turn_on()
        else:
            host.turn_off()


class Controller:
    def __init__(self, wf, hosts, rates, args):
        self.wf, self.hosts, self.args = wf, hosts, args
        self.rates = {h: rates[h.name] if args.assume_rate is None else args.assume_rate for h in hosts}
        self.up, _ = ranks(wf)
        self.weight = criticality(wf, args.criticality)
        self.index = {t: i for i, t in enumerate(wf["order"])}
        self.missing = {t: len(wf["parents"][t]) for t in wf["order"]}
        self.ready = {t for t in wf["order"] if self.missing[t] == 0}
        self.where = {}                                    # task -> host holding its outputs
        self.busy = dict.fromkeys(hosts)                   # host -> task running on it, or None
        self.attempt = {}                                  # task -> (exec, [comms], host)
        self.pending = ActivitySet()
        self.stats = {"reruns": 0, "wasted_s": 0.0, "transfer_retries": 0}

    def eft(self, task, host, now):
        inputs = (transfer_time(self.wf["edge"][p, task]) for p in self.wf["parents"][task] if self.where[p] != host)
        return now + max(inputs, default=0.0) + self.wf["flops"][task] / host.speed

    def score(self, task, host, now):
        finish = self.eft(task, host, now)
        if self.args.policy != "risk":
            return finish
        exposure = finish - now
        p_fail = 1.0 - math.exp(-self.rates[host] * exposure)
        return finish + (1.0 + self.args.lam * self.weight[task]) * p_fail * exposure

    @staticmethod
    def best(candidates):
        """Lowest score; scores within 1e-9 (relative) count as equal and the first candidate wins."""
        best = None
        for value, item in candidates:
            if best is None or value < best[0] - 1e-9 * max(1.0, abs(best[0])):
                best = (value, item)
        return best

    def schedule(self):
        now = Engine.clock
        idle = [h for h in self.hosts if h.is_on and self.busy[h] is None]
        tasks = [t for t in self.ready if all(self.where[p].is_on for p in self.wf["parents"][t])]
        if self.args.policy == "minmin":
            tasks.sort(key=self.index.get)
            while tasks and idle:
                _, (t, h) = self.best((self.eft(t, h, now), (t, h)) for t in tasks for h in idle)
                self.start(t, h)
                tasks.remove(t)
                idle.remove(h)
            return
        tasks.sort(key=lambda t: (-self.up[t], self.index[t]))
        for t in tasks:
            if not idle:
                break
            _, h = self.best((self.score(t, h, now), h) for h in idle)
            self.start(t, h)
            idle.remove(h)

    def start(self, task, host):
        execution = Exec.init().set_flops_amount(self.wf["flops"][task])
        execution.name = task
        execution.host = host
        comms = []
        for p in self.wf["parents"][task]:
            nbytes = self.wf["edge"][p, task]
            if self.where[p] != host and nbytes > 0:
                comm = Comm.sendto_init(self.where[p], host).set_payload_size(int(nbytes))
                comm.add_successor(execution)
                comms.append(comm)
        for activity in comms + [execution]:
            activity.start()
            self.pending.push(activity)
        self.ready.discard(task)
        self.busy[host] = task
        self.attempt[task] = (execution, comms, host)

    def abort(self, task, failed):
        """Undo an attempt after one of its activities failed, and put the task back in the ready set."""
        execution, comms, host = self.attempt.pop(task)
        if failed is execution:
            self.stats["reruns"] += 1
            started = execution.start_time                 # SimGrid resets 'remaining' on failure, so use the clock
            self.stats["wasted_s"] += max(0.0, Engine.clock - started) if 0 <= started <= Engine.clock else 0.0
            if not self.args.json:
                this_actor.info(f"'{task}' failed on {host.name}: placing it again")
        else:
            self.stats["transfer_retries"] += 1
        for activity in comms + [execution]:
            if activity is not failed and activity.state_str not in ("FINISHED", "FAILED", "CANCELED"):
                self.pending.erase(activity)
                activity.cancel()
        self.busy[host] = None
        self.ready.add(task)

    def owner(self, activity):
        for task, (execution, comms, _) in self.attempt.items():
            if activity is execution or any(activity is c for c in comms):
                return task
        return None

    def run(self):
        left = len(self.wf["order"])
        while left > 0 and Engine.clock < self.args.horizon:
            self.schedule()
            if self.pending.empty():                       # every task waits for a host to come back
                this_actor.sleep_for(self.args.poll)
                continue
            try:
                if any(not h.is_on for h in self.hosts):
                    done = self.pending.wait_any_for(self.args.poll)
                else:
                    done = self.pending.wait_any()
            except TimeoutException:
                continue
            except (HostFailureException, NetworkFailureException):
                failed = self.pending.get_failed_activity()
                while failed is not None:
                    task = self.owner(failed)
                    if task is not None:
                        self.abort(task, failed)
                    failed = self.pending.get_failed_activity()
                continue
            task = self.owner(done)
            if task is None or done is not self.attempt[task][0]:
                continue                                   # a transfer completed; its execution starts by itself
            _, _, host = self.attempt.pop(task)
            self.busy[host] = None
            self.where[task] = host
            left -= 1
            for c in self.wf["children"][task]:
                self.missing[c] -= 1
                if self.missing[c] == 0:
                    self.ready.add(c)
        sg_host_energy_update_all()
        self.stats.update(makespan=Engine.clock, completed=left == 0,
                          energy_J=sum(h.consumed_energy for h in self.hosts))


def main():
    e = Engine(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow")
    parser.add_argument("--policy", choices=["minmin", "heft", "risk"], default="heft")
    parser.add_argument("--criticality", default="uniform",
                        choices=["uniform", "rank", "descendants", "critical-path", "centrality", "entropy",
                                 "dependency"])
    parser.add_argument("--lam", type=float, default=2.0, help="weight of the failure penalty (risk policy)")
    parser.add_argument("--rate", type=float, default=0.0, help="failures per second of a normal host")
    parser.add_argument("--downtime", type=float, default=30.0, help="seconds a failed host stays off")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=float, default=1e6, help="give up at this simulated time")
    parser.add_argument("--poll", type=float, default=1.0, help="re-check interval while a host is off")
    parser.add_argument("--assume-rate", type=float, help="failure rate the risk policy assumes for every host "
                        "(default: each host's real rate)")
    parser.add_argument("--json", action="store_true", help="print a JSON summary instead of the log")
    args, _ = parser.parse_known_args([a for a in sys.argv[1:] if not a.startswith(("--cfg", "--log"))])

    sg_host_energy_plugin_init()
    wf = load_workflow(args.workflow)
    ctl, hosts = build_platform(e)
    rates, events = failure_schedule(hosts, args.rate, args.downtime, args.seed, args.horizon)
    controller = Controller(wf, hosts, rates, args)
    ctl.add_actor("controller", controller.run)
    if events:
        ctl.add_actor("chaos", chaos, events).daemonize()
    e.run()

    stats = controller.stats
    if args.json:
        print(json.dumps({**stats, "hosts": [h.name for h in e.all_hosts], "tasks": len(wf["order"])}))
    else:
        this_actor.info(f"Makespan {stats['makespan']:.2f} s, {stats['reruns']} re-runs "
                        f"({stats['wasted_s']:.2f} s lost), {stats['transfer_retries']} transfer retries, "
                        f"{stats['energy_J']:.0f} J")


if __name__ == "__main__":
    main()
