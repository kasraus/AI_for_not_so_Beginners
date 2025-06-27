#!/usr/bin/env python3
"""
Android Trace Task Dependency Analyzer

This script analyzes Android system traces (atrace, ftrace, systrace) to:
1. Extract all tasks (processes and threads)
2. Identify inter-task dependencies
3. Create a task dependency graph visualization

Supports multiple trace formats and dependency detection methods.
"""

import re
import json
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, namedtuple
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Task:
    """Represents a process or thread in the system"""
    pid: int
    tid: int
    name: str
    tgid: int = None
    is_thread: bool = False
    
    def __hash__(self):
        return hash((self.pid, self.tid))
    
    def __str__(self):
        if self.is_thread:
            return f"{self.name}({self.pid}:{self.tid})"
        return f"{self.name}({self.pid})"

@dataclass
class TraceEvent:
    """Represents a single trace event"""
    timestamp: float
    task: Task
    event_type: str
    details: Dict
    cpu: int = None

class DependencyType:
    """Types of dependencies between tasks"""
    IPC_BINDER = "binder_ipc"
    SIGNAL = "signal"
    FUTEX = "futex"
    PIPE = "pipe"
    SOCKET = "socket"
    SHARED_MEMORY = "shared_mem"
    PARENT_CHILD = "parent_child"
    WAKEUP = "wakeup"
    MUTEX = "mutex"
    CONDVAR = "condvar"

class AndroidTraceAnalyzer:
    """Main analyzer class for Android traces"""
    
    def __init__(self):
        self.tasks: Dict[Tuple[int, int], Task] = {}
        self.events: List[TraceEvent] = []
        self.dependencies: Dict[Tuple[Task, Task], Set[str]] = defaultdict(set)
        self.graph = nx.DiGraph()
        
        # Regex patterns for different trace formats
        self.patterns = {
            'atrace': re.compile(r'^\s*(\S+)-(\d+)\s+\[(\d+)\]\s+([d\.]{4})\s+(\d+\.\d+):\s+(.+)$'),
            'ftrace': re.compile(r'^\s*(\S+)-(\d+)\s+\[(\d+)\]\s+([d\.]{4})\s+(\d+\.\d+):\s+(.+)$'),
            'systrace_html': re.compile(r'"ts":(\d+\.\d+).*?"pid":(\d+).*?"tid":(\d+).*?"name":"([^"]*)"'),
            'binder_transaction': re.compile(r'binder_transaction:\s+transaction=(\d+)\s+dest_node=(\d+)\s+dest_proc=(\d+)\s+dest_thread=(\d+)'),
            'sched_wakeup': re.compile(r'sched_wakeup:\s+comm=(\S+)\s+pid=(\d+)\s+prio=(\d+)\s+success=(\d+)\s+target_cpu=(\d+)'),
            'sched_switch': re.compile(r'sched_switch:\s+prev_comm=(\S+)\s+prev_pid=(\d+).*?next_comm=(\S+)\s+next_pid=(\d+)'),
            'signal_deliver': re.compile(r'signal_deliver:\s+sig=(\d+)\s+errno=(\d+)\s+code=(\d+)\s+sa_handler=([0-9a-fA-Fx]+)'),
            'sys_futex': re.compile(r'sys_futex.*?\(uaddr:\s*([0-9a-fA-Fx]+)'),
        }
    
    def parse_trace_file(self, filepath: str) -> None:
        """Parse trace file and extract events"""
        print(f"Parsing trace file: {filepath}")
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Detect trace format
        if filepath.endswith('.html') or 'tracing_mark_write' in content:
            self._parse_systrace_html(content)
        else:
            self._parse_ftrace_format(content)
    
    def _parse_ftrace_format(self, content: str) -> None:
        """Parse ftrace/atrace format"""
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            match = self.patterns['atrace'].match(line)
            if not match:
                continue
            
            try:
                comm, pid, cpu, flags, timestamp, event_data = match.groups()
                pid = int(pid)
                cpu = int(cpu)
                timestamp = float(timestamp)
                
                # Extract TID from event data if available
                tid = pid  # Default to PID
                tid_match = re.search(r'pid=(\d+)', event_data)
                if tid_match:
                    tid = int(tid_match.group(1))
                
                # Create or get task
                task = self._get_or_create_task(pid, tid, comm)
                
                # Parse specific event types
                event_type, details = self._parse_event_data(event_data)
                
                event = TraceEvent(
                    timestamp=timestamp,
                    task=task,
                    event_type=event_type,
                    details=details,
                    cpu=cpu
                )
                
                self.events.append(event)
                self._analyze_dependencies(event)
                
            except (ValueError, IndexError) as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    def _parse_systrace_html(self, content: str) -> None:
        """Parse systrace HTML format"""
        # Extract JSON data from HTML
        json_match = re.search(r'var trace_data = ({.*?});', content, re.DOTALL)
        if not json_match:
            print("No trace data found in HTML file")
            return
        
        try:
            trace_data = json.loads(json_match.group(1))
            events = trace_data.get('traceEvents', [])
            
            for event in events:
                if 'ts' not in event or 'pid' not in event:
                    continue
                
                timestamp = event['ts'] / 1000000.0  # Convert to seconds
                pid = event['pid']
                tid = event.get('tid', pid)
                name = event.get('name', 'unknown')
                
                task = self._get_or_create_task(pid, tid, name)
                
                trace_event = TraceEvent(
                    timestamp=timestamp,
                    task=task,
                    event_type=event.get('ph', 'unknown'),
                    details=event
                )
                
                self.events.append(trace_event)
                
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
    
    def _get_or_create_task(self, pid: int, tid: int, name: str) -> Task:
        """Get existing task or create new one"""
        key = (pid, tid)
        if key not in self.tasks:
            is_thread = pid != tid
            task = Task(pid=pid, tid=tid, name=name, is_thread=is_thread)
            self.tasks[key] = task
        return self.tasks[key]
    
    def _parse_event_data(self, event_data: str) -> Tuple[str, Dict]:
        """Parse event data to extract type and details"""
        details = {}
        
        # Binder transaction
        if 'binder_transaction' in event_data:
            match = self.patterns['binder_transaction'].search(event_data)
            if match:
                details = {
                    'transaction_id': int(match.group(1)),
                    'dest_node': int(match.group(2)),
                    'dest_proc': int(match.group(3)),
                    'dest_thread': int(match.group(4))
                }
                return DependencyType.IPC_BINDER, details
        
        # Scheduler wakeup
        if 'sched_wakeup' in event_data:
            match = self.patterns['sched_wakeup'].search(event_data)
            if match:
                details = {
                    'target_comm': match.group(1),
                    'target_pid': int(match.group(2)),
                    'priority': int(match.group(3)),
                    'success': int(match.group(4)),
                    'target_cpu': int(match.group(5))
                }
                return DependencyType.WAKEUP, details
        
        # Scheduler switch
        if 'sched_switch' in event_data:
            match = self.patterns['sched_switch'].search(event_data)
            if match:
                details = {
                    'prev_comm': match.group(1),
                    'prev_pid': int(match.group(2)),
                    'next_comm': match.group(3),
                    'next_pid': int(match.group(4))
                }
                return 'sched_switch', details
        
        # Signal delivery
        if 'signal_deliver' in event_data:
            match = self.patterns['signal_deliver'].search(event_data)
            if match:
                details = {
                    'signal': int(match.group(1)),
                    'errno': int(match.group(2)),
                    'code': int(match.group(3)),
                    'handler': match.group(4)
                }
                return DependencyType.SIGNAL, details
        
        # Futex operations
        if 'sys_futex' in event_data:
            match = self.patterns['sys_futex'].search(event_data)
            if match:
                details = {'futex_addr': match.group(1)}
                return DependencyType.FUTEX, details
        
        return 'generic', {'raw': event_data}
    
    def _analyze_dependencies(self, event: TraceEvent) -> None:
        """Analyze event for task dependencies"""
        if event.event_type == DependencyType.IPC_BINDER:
            self._handle_binder_dependency(event)
        elif event.event_type == DependencyType.WAKEUP:
            self._handle_wakeup_dependency(event)
        elif event.event_type == DependencyType.SIGNAL:
            self._handle_signal_dependency(event)
        elif event.event_type == DependencyType.FUTEX:
            self._handle_futex_dependency(event)
    
    def _handle_binder_dependency(self, event: TraceEvent) -> None:
        """Handle binder IPC dependencies"""
        details = event.details
        if 'dest_proc' in details:
            dest_pid = details['dest_proc']
            dest_tid = details.get('dest_thread', dest_pid)
            
            # Find or create destination task
            dest_task = None
            for task in self.tasks.values():
                if task.pid == dest_pid and task.tid == dest_tid:
                    dest_task = task
                    break
            
            if dest_task:
                self.dependencies[(event.task, dest_task)].add(DependencyType.IPC_BINDER)
    
    def _handle_wakeup_dependency(self, event: TraceEvent) -> None:
        """Handle wakeup dependencies"""
        details = event.details
        if 'target_pid' in details:
            target_pid = details['target_pid']
            
            # Find target task
            target_task = None
            for task in self.tasks.values():
                if task.pid == target_pid:
                    target_task = task
                    break
            
            if target_task:
                self.dependencies[(event.task, target_task)].add(DependencyType.WAKEUP)
    
    def _handle_signal_dependency(self, event: TraceEvent) -> None:
        """Handle signal dependencies"""
        # Signal dependencies are typically from sender to receiver
        # This would need more context from the trace to determine sender
        pass
    
    def _handle_futex_dependency(self, event: TraceEvent) -> None:
        """Handle futex synchronization dependencies"""
        # Group tasks using same futex address
        futex_addr = event.details.get('futex_addr')
        if futex_addr:
            # Find other tasks using same futex
            for other_event in self.events:
                if (other_event.event_type == DependencyType.FUTEX and 
                    other_event.details.get('futex_addr') == futex_addr and
                    other_event.task != event.task):
                    self.dependencies[(event.task, other_event.task)].add(DependencyType.FUTEX)
    
    def detect_parent_child_relationships(self) -> None:
        """Detect parent-child process relationships"""
        # This would typically require parsing process creation events
        # or using additional system information
        process_pids = set(task.pid for task in self.tasks.values() if not task.is_thread)
        
        # Simple heuristic: processes with similar names might be related
        processes_by_name = defaultdict(list)
        for task in self.tasks.values():
            if not task.is_thread:
                base_name = re.sub(r'\d+$', '', task.name)
                processes_by_name[base_name].append(task)
        
        # Add parent-child relationships for processes with same base name
        for name, procs in processes_by_name.items():
            if len(procs) > 1:
                procs.sort(key=lambda x: x.pid)
                for i in range(len(procs) - 1):
                    parent = procs[i]
                    child = procs[i + 1]
                    self.dependencies[(parent, child)].add(DependencyType.PARENT_CHILD)
    
    def build_dependency_graph(self) -> None:
        """Build NetworkX graph from dependencies"""
        self.graph.clear()
        
        # Add all tasks as nodes
        for task in self.tasks.values():
            self.graph.add_node(task, label=str(task))
        
        # Add dependency edges
        for (source, dest), dep_types in self.dependencies.items():
            edge_label = ','.join(dep_types)
            self.graph.add_edge(source, dest, 
                              dependency_types=list(dep_types),
                              label=edge_label,
                              weight=len(dep_types))
    
    def analyze_critical_path(self) -> List[Task]:
        """Find critical path in dependency graph"""
        try:
            # Find longest path (critical path)
            longest_path = nx.dag_longest_path(self.graph)
            return longest_path
        except nx.NetworkXError:
            # Graph has cycles, try to find strongly connected components
            sccs = list(nx.strongly_connected_components(self.graph))
            if sccs:
                return list(max(sccs, key=len))
            return []
    
    def generate_report(self) -> Dict:
        """Generate analysis report"""
        report = {
            'summary': {
                'total_tasks': len(self.tasks),
                'total_processes': len([t for t in self.tasks.values() if not t.is_thread]),
                'total_threads': len([t for t in self.tasks.values() if t.is_thread]),
                'total_dependencies': len(self.dependencies),
                'total_events': len(self.events)
            },
            'dependency_types': {},
            'critical_path': [],
            'top_communicators': [],
            'isolated_tasks': []
        }
        
        # Count dependency types
        all_dep_types = []
        for dep_set in self.dependencies.values():
            all_dep_types.extend(dep_set)
        
        for dep_type in set(all_dep_types):
            report['dependency_types'][dep_type] = all_dep_types.count(dep_type)
        
        # Find critical path
        critical_path = self.analyze_critical_path()
        report['critical_path'] = [str(task) for task in critical_path]
        
        # Find top communicators (nodes with most connections)
        node_degrees = dict(self.graph.degree())
        top_comm = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        report['top_communicators'] = [(str(task), degree) for task, degree in top_comm]
        
        # Find isolated tasks
        isolated = [task for task in self.tasks.values() if self.graph.degree(task) == 0]
        report['isolated_tasks'] = [str(task) for task in isolated]
        
        return report
    
    def visualize_graph(self, output_file: str = 'task_dependency_graph.png', 
                       max_nodes: int = 50) -> None:
        """Visualize the dependency graph"""
        if len(self.graph.nodes()) > max_nodes:
            print(f"Graph has {len(self.graph.nodes())} nodes, showing top {max_nodes} by degree")
            # Show only most connected nodes
            node_degrees = dict(self.graph.degree())
            top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            subgraph_nodes = [node for node, _ in top_nodes]
            G = self.graph.subgraph(subgraph_nodes)
        else:
            G = self.graph
        
        plt.figure(figsize=(16, 12))
        
        # Use hierarchical layout for better visualization
        try:
            pos = nx.spring_layout(G, k=2, iterations=50)
        except:
            pos = nx.random_layout(G)
        
        # Draw nodes
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            if node.is_thread:
                node_colors.append('lightblue')
                node_sizes.append(300)
            else:
                node_colors.append('lightcoral')
                node_sizes.append(500)
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.7)
        
        # Draw edges with different colors for different dependency types
        edge_colors = {
            DependencyType.IPC_BINDER: 'red',
            DependencyType.WAKEUP: 'blue',
            DependencyType.SIGNAL: 'green',
            DependencyType.FUTEX: 'orange',
            DependencyType.PARENT_CHILD: 'purple'
        }
        
        for dep_type, color in edge_colors.items():
            edges = [(u, v) for u, v, d in G.edges(data=True) 
                    if dep_type in d.get('dependency_types', [])]
            if edges:
                nx.draw_networkx_edges(G, pos, edgelist=edges, 
                                     edge_color=color, alpha=0.6, width=2)
        
        # Draw labels
        labels = {node: node.name[:10] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title('Android Task Dependency Graph', fontsize=16)
        plt.legend(handles=[plt.Line2D([0], [0], color=color, lw=2, label=dep_type) 
                           for dep_type, color in edge_colors.items()], 
                  loc='upper right')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Graph saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze Android traces for task dependencies')
    parser.add_argument('trace_file', help='Path to trace file (atrace, ftrace, or systrace)')
    parser.add_argument('--output', '-o', default='analysis_report.json', 
                       help='Output file for analysis report')
    parser.add_argument('--graph', '-g', default='task_graph.png', 
                       help='Output file for dependency graph visualization')
    parser.add_argument('--max-nodes', type=int, default=50,
                       help='Maximum nodes to show in graph visualization')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip graph visualization')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.trace_file).exists():
        print(f"Error: Trace file '{args.trace_file}' not found")
        return 1
    
    # Initialize analyzer
    analyzer = AndroidTraceAnalyzer()
    
    try:
        # Parse trace file
        analyzer.parse_trace_file(args.trace_file)
        
        # Detect additional relationships
        analyzer.detect_parent_child_relationships()
        
        # Build dependency graph
        analyzer.build_dependency_graph()
        
        # Generate report
        report = analyzer.generate_report()
        
        # Save report
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Analysis report saved to {args.output}")
        
        # Print summary
        print("\n=== ANALYSIS SUMMARY ===")
        print(f"Total Tasks: {report['summary']['total_tasks']}")
        print(f"  Processes: {report['summary']['total_processes']}")
        print(f"  Threads: {report['summary']['total_threads']}")
        print(f"Dependencies: {report['summary']['total_dependencies']}")
        print(f"Events Processed: {report['summary']['total_events']}")
        
        print("\nDependency Types:")
        for dep_type, count in report['dependency_types'].items():
            print(f"  {dep_type}: {count}")
        
        if report['critical_path']:
            print(f"\nCritical Path ({len(report['critical_path'])} tasks):")
            for task in report['critical_path'][:5]:  # Show first 5
                print(f"  {task}")
            if len(report['critical_path']) > 5:
                print(f"  ... and {len(report['critical_path']) - 5} more")
        
        # Visualize graph
        if not args.no_viz:
            analyzer.visualize_graph(args.graph, args.max_nodes)
        
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1

if __name__ == '__main__':
    exit(main())