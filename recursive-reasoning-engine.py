"""
Recursive Reasoning Engine Core
===============================

This module implements the core components of the Recursive Reasoning Engine,
providing mechanisms for:
1. Building recursive scaffolds from reasoning attempts
2. Detecting and managing reasoning loops
3. Tracking symbolic residue from reasoning failures
4. Representing and analyzing recursive reasoning structures as graphs

The engine is designed to transform linear chain-of-thought reasoning into
recursive pattern-based scaffolds that support co-emergent reasoning between
humans and AI models.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ======================================================
# Data Structures for Recursive Reasoning
# ======================================================

class ReasoningNodeType(Enum):
    """Types of nodes in a reasoning scaffold"""
    PREMISE = "premise"            # Initial given information
    SUBGOAL = "subgoal"            # Intermediate reasoning step
    INFERENCE = "inference"        # Logical deduction
    OBSERVATION = "observation"    # Direct observation
    REFLECTION = "reflection"      # Meta-reasoning
    QUESTION = "question"          # Query or uncertainty
    CONTRADICTION = "contradiction" # Identified conflict
    RESOLUTION = "resolution"      # Resolution of contradiction
    CONCLUSION = "conclusion"      # Final outcome
    META = "meta"                  # Meta-reasoning about the reasoning process
    ECHO = "echo"                  # Recursive reference to another node
    RESIDUE = "residue"            # Symbolic residue from a failure


class ReasoningEdgeType(Enum):
    """Types of edges in a reasoning scaffold"""
    DEDUCTION = "deduction"        # Logical deduction
    ABDUCTION = "abduction"        # Inference to best explanation
    INDUCTION = "induction"        # Pattern generalization
    ANALOGY = "analogy"            # Similarity-based reasoning
    RECURSION = "recursion"        # Recursive reference
    REFLECTION = "reflection"      # Meta-cognitive reflection
    CONTRADICTION = "contradiction" # Logical conflict
    REVISION = "revision"          # Update based on new information
    QUESTION = "question"          # Question about node
    RESOLUTION = "resolution"      # Resolution of contradiction


@dataclass
class SymbolicResidue:
    """Represents the symbolic residue of a reasoning attempt"""
    content: str                   # The content of the residue
    source_node_id: Optional[str]  # ID of the node that generated the residue
    pattern_signature: str         # Signature identifying the pattern type
    confidence: float              # Confidence in the residue (0.0-1.0)
    origin_type: ReasoningNodeType # Type of node that generated the residue
    failure_context: Dict[str, Any] # Context of the failure


@dataclass
class ReasoningNode:
    """Represents a node in a reasoning scaffold"""
    id: str
    content: str
    node_type: ReasoningNodeType
    metadata: Dict[str, Any]
    created_by: str                # "human", "model", or "co-emergent"
    confidence: float              # Confidence in the node (0.0-1.0)
    residue: Optional[SymbolicResidue] = None  # Associated symbolic residue


@dataclass
class ReasoningEdge:
    """Represents an edge in a reasoning scaffold"""
    source_id: str
    target_id: str
    edge_type: ReasoningEdgeType
    weight: float                  # Strength of the connection
    metadata: Dict[str, Any]
    created_by: str                # "human", "model", or "co-emergent"


class RecursiveScaffold:
    """
    Represents a recursive reasoning scaffold as a directed graph.
    
    The scaffold consists of reasoning nodes connected by typed edges,
    forming a structure that supports recursive, non-linear reasoning.
    """
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, ReasoningNode] = {}
        self.failed_attempts: List[Tuple[ReasoningNode, SymbolicResidue]] = []
        self.loop_history: List[List[str]] = []
        self.metadata: Dict[str, Any] = {
            "creation_timestamp": None,
            "last_modified": None,
            "contributors": [],
            "domain": None,
            "recursion_depth": 0
        }
    
    def add_node(self, node: ReasoningNode) -> str:
        """Add a node to the scaffold"""
        if node.id in self.nodes:
            logger.warning(f"Node with ID {node.id} already exists. Updating instead.")
        
        self.nodes[node.id] = node
        self.graph.add_node(node.id, 
                           node_type=node.node_type.value,
                           content=node.content,
                           metadata=node.metadata,
                           created_by=node.created_by,
                           confidence=node.confidence)
        return node.id
    
    def add_edge(self, edge: ReasoningEdge) -> None:
        """Add an edge between nodes in the scaffold"""
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            raise ValueError(f"Source or target node does not exist: {edge.source_id} -> {edge.target_id}")
        
        self.graph.add_edge(edge.source_id, edge.target_id,
                           edge_type=edge.edge_type.value,
                           weight=edge.weight,
                           metadata=edge.metadata,
                           created_by=edge.created_by)
    
    def record_failed_attempt(self, node: ReasoningNode, residue: SymbolicResidue) -> None:
        """Record a failed reasoning attempt and its symbolic residue"""
        self.failed_attempts.append((node, residue))
        node.residue = residue
        self.add_node(node)
    
    def detect_loops(self) -> List[List[str]]:
        """Detect reasoning loops in the scaffold"""
        cycles = list(nx.simple_cycles(self.graph))
        self.loop_history.extend(cycles)
        return cycles
    
    def get_recursion_paths(self) -> List[List[str]]:
        """Find paths that involve recursive reasoning"""
        recursion_edges = [(s, t) for s, t, d in self.graph.edges(data=True) 
                         if d.get('edge_type') == ReasoningEdgeType.RECURSION.value]
        
        recursion_paths = []
        for source, target in recursion_edges:
            for path in nx.all_simple_paths(self.graph, source=target, target=source):
                recursion_paths.append(path + [source])  # Complete the loop
        
        return recursion_paths
    
    def calculate_recursion_depth(self) -> int:
        """Calculate the maximum recursion depth in the scaffold"""
        recursion_paths = self.get_recursion_paths()
        if not recursion_paths:
            return 0
        
        max_depth = max(len(path) for path in recursion_paths)
        self.metadata["recursion_depth"] = max_depth
        return max_depth
    
    def extract_reflection_subgraph(self) -> nx.DiGraph:
        """Extract the subgraph of reflection nodes"""
        reflection_nodes = [n for n, d in self.graph.nodes(data=True) 
                          if d.get('node_type') == ReasoningNodeType.REFLECTION.value]
        return self.graph.subgraph(reflection_nodes)
    
    def find_contradictions(self) -> List[Tuple[str, str]]:
        """Find contradiction edges in the scaffold"""
        contradiction_edges = [(s, t) for s, t, d in self.graph.edges(data=True) 
                             if d.get('edge_type') == ReasoningEdgeType.CONTRADICTION.value]
        return contradiction_edges
    
    def get_residue_patterns(self) -> Dict[str, List[SymbolicResidue]]:
        """Group symbolic residue by pattern signature"""
        patterns = {}
        for node in self.nodes.values():
            if node.residue:
                signature = node.residue.pattern_signature
                if signature not in patterns:
                    patterns[signature] = []
                patterns[signature].append(node.residue)
        return patterns
    
    def to_dot(self) -> str:
        """Convert scaffold to DOT format for visualization"""
        # Implementation depends on specific visualization requirements
        pass
    
    def to_dict(self) -> Dict:
        """Convert scaffold to dictionary for serialization"""
        return {
            "name": self.name,
            "description": self.description,
            "nodes": {node_id: {
                "id": node.id,
                "content": node.content,
                "node_type": node.node_type.value,
                "metadata": node.metadata,
                "created_by": node.created_by,
                "confidence": node.confidence,
                "residue": None if node.residue is None else {
                    "content": node.residue.content,
                    "source_node_id": node.residue.source_node_id,
                    "pattern_signature": node.residue.pattern_signature,
                    "confidence": node.residue.confidence,
                    "origin_type": node.residue.origin_type.value,
                    "failure_context": node.residue.failure_context
                }
            } for node_id, node in self.nodes.items()},
            "edges": [{
                "source_id": s,
                "target_id": t,
                "edge_type": d["edge_type"],
                "weight": d["weight"],
                "metadata": d["metadata"],
                "created_by": d["created_by"]
            } for s, t, d in self.graph.edges(data=True)],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RecursiveScaffold':
        """Create scaffold from dictionary representation"""
        scaffold = cls(data["name"], data["description"])
        scaffold.metadata = data["metadata"]
        
        # Add nodes
        for node_data in data["nodes"].values():
            residue = None
            if node_data.get("residue"):
                r = node_data["residue"]
                residue = SymbolicResidue(
                    content=r["content"],
                    source_node_id=r["source_node_id"],
                    pattern_signature=r["pattern_signature"],
                    confidence=r["confidence"],
                    origin_type=ReasoningNodeType(r["origin_type"]),
                    failure_context=r["failure_context"]
                )
            
            node = ReasoningNode(
                id=node_data["id"],
                content=node_data["content"],
                node_type=ReasoningNodeType(node_data["node_type"]),
                metadata=node_data["metadata"],
                created_by=node_data["created_by"],
                confidence=node_data["confidence"],
                residue=residue
            )
            scaffold.add_node(node)
        
        # Add edges
        for edge_data in data["edges"]:
            edge = ReasoningEdge(
                source_id=edge_data["source_id"],
                target_id=edge_data["target_id"],
                edge_type=ReasoningEdgeType(edge_data["edge_type"]),
                weight=edge_data["weight"],
                metadata=edge_data["metadata"],
                created_by=edge_data["created_by"]
            )
            scaffold.add_edge(edge)
        
        return scaffold


# ======================================================
# Scaffold Builder
# ======================================================

class ScaffoldBuilder:
    """
    Builds recursive reasoning scaffolds from reasoning attempts.
    
    The builder analyzes reasoning attempts (successful and failed),
    extracts patterns, and constructs a scaffold that supports
    recursive reasoning on the given problem.
    """
    
    def __init__(self):
        self.pattern_library = {}  # Library of known reasoning patterns
    
    def create_scaffold(self, name: str, description: str) -> RecursiveScaffold:
        """Create a new recursive scaffold"""
        return RecursiveScaffold(name, description)
    
    def extract_from_chain_of_thought(self, cot_text: str, created_by: str) -> RecursiveScaffold:
        """Extract a scaffold from a chain-of-thought reasoning text"""
        # Implementation would parse the CoT and extract nodes/edges
        scaffold = RecursiveScaffold(f"Scaffold from CoT", "Extracted from chain-of-thought reasoning")
        
        # Simple parsing logic - this would be more sophisticated in practice
        steps = cot_text.split("\n")
        prev_node_id = None
        
        for i, step in enumerate(steps):
            if not step.strip():
                continue
                
            # Create node for this step
            node_id = f"node_{i}"
            node_type = ReasoningNodeType.INFERENCE
            if i == 0:
                node_type = ReasoningNodeType.PREMISE
            elif i == len(steps) - 1:
                node_type = ReasoningNodeType.CONCLUSION
                
            node = ReasoningNode(
                id=node_id,
                content=step,
                node_type=node_type,
                metadata={"step": i},
                created_by=created_by,
                confidence=0.9  # Default confidence
            )
            scaffold.add_node(node)
            
            # Connect to previous node
            if prev_node_id:
                edge = ReasoningEdge(
                    source_id=prev_node_id,
                    target_id=node_id,
                    edge_type=ReasoningEdgeType.DEDUCTION,
                    weight=1.0,
                    metadata={},
                    created_by=created_by
                )
                scaffold.add_edge(edge)
            
            prev_node_id = node_id
        
        return scaffold
    
    def identify_loop_opportunities(self, scaffold: RecursiveScaffold) -> List[Tuple[str, str]]:
        """Identify opportunities for reasoning loops in the scaffold"""
        potential_loops = []
        
        # Look for nodes with similar content or pattern
        for node1_id, node1 in scaffold.nodes.items():
            for node2_id, node2 in scaffold.nodes.items():
                if node1_id != node2_id:
                    # Compute similarity - this would use more sophisticated measures in practice
                    similarity = self._calculate_node_similarity(node1, node2)
                    if similarity > 0.7:  # Threshold for potential loop
                        potential_loops.append((node1_id, node2_id))
        
        return potential_loops
    
    def _calculate_node_similarity(self, node1: ReasoningNode, node2: ReasoningNode) -> float:
        """Calculate similarity between two reasoning nodes"""
        # This would use embedding similarity, pattern matching, etc.
        # Simple implementation for demonstration
        words1 = set(node1.content.lower().split())
        words2 = set(node2.content.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        return len(intersection) / min(len(words1), len(words2))
    
    def add_reflection_layer(self, scaffold: RecursiveScaffold, reflection: str, 
                           target_node_ids: List[str], created_by: str) -> str:
        """Add a reflection layer to the scaffold"""
        reflection_id = f"reflection_{len([n for n in scaffold.nodes.values() if n.node_type == ReasoningNodeType.REFLECTION])}"
        
        reflection_node = ReasoningNode(
            id=reflection_id,
            content=reflection,
            node_type=ReasoningNodeType.REFLECTION,
            metadata={"targets": target_node_ids},
            created_by=created_by,
            confidence=0.8
        )
        
        scaffold.add_node(reflection_node)
        
        # Connect reflection to target nodes
        for target_id in target_node_ids:
            edge = ReasoningEdge(
                source_id=reflection_id,
                target_id=target_id,
                edge_type=ReasoningEdgeType.REFLECTION,
                weight=1.0,
                metadata={},
                created_by=created_by
            )
            scaffold.add_edge(edge)
        
        return reflection_id
    
    def integrate_failed_attempt(self, scaffold: RecursiveScaffold, 
                                failed_content: str, 
                                failure_context: Dict[str, Any],
                                created_by: str) -> str:
        """Integrate a failed reasoning attempt into the scaffold"""
        # Create a node for the failed attempt
        failed_node_id = f"failed_{len(scaffold.failed_attempts)}"
        
        # Determine the pattern signature of this failure
        pattern_signature = self._analyze_failure_pattern(failed_content, failure_context)
        
        # Create symbolic residue
        residue = SymbolicResidue(
            content=failed_content,
            source_node_id=None,  # No source yet
            pattern_signature=pattern_signature,
            confidence=0.7,  # Initial confidence in this pattern
            origin_type=ReasoningNodeType.CONTRADICTION,  # Default to contradiction
            failure_context=failure_context
        )
        
        # Create node for failed attempt
        failed_node = ReasoningNode(
            id=failed_node_id,
            content=failed_content,
            node_type=ReasoningNodeType.RESIDUE,
            metadata={"pattern": pattern_signature},
            created_by=created_by,
            confidence=0.5,  # Lower confidence since it's a failure
            residue=residue
        )
        
        # Record the failed attempt
        scaffold.record_failed_attempt(failed_node, residue)
        
        # Try to connect it to existing nodes
        connected = self._connect_failed_node(scaffold, failed_node_id)
        
        return failed_node_id
    
    def _analyze_failure_pattern(self, content: str, context: Dict[str, Any]) -> str:
        """Analyze the pattern of a reasoning failure"""
        # This would use more sophisticated pattern analysis in practice
        # For now, we'll use a simple keyword-based approach
        
        pattern_signatures = {
            "circular": ["circular", "loop", "same as", "repeating"],
            "contradiction": ["contradiction", "inconsistent", "cannot be both", "conflict"],
            "incomplete": ["incomplete", "missing", "not enough", "need more"],
            "invalid": ["invalid", "incorrect", "wrong", "error"],
            "overspecific": ["too specific", "special case", "doesn't generalize"],
            "underspecific": ["too general", "vague", "ambiguous"],
        }
        
        # Check for pattern matches
        for pattern, keywords in pattern_signatures.items():
            if any(keyword in content.lower() for keyword in keywords):
                return pattern
        
        # Default pattern if no match
        return "unclassified"
    
    def _connect_failed_node(self, scaffold: RecursiveScaffold, failed_node_id: str) -> bool:
        """Try to connect a failed node to existing nodes in the scaffold"""
        connected = False
        failed_node = scaffold.nodes[failed_node_id]
        
        # Find potential connections based on content similarity
        for node_id, node in scaffold.nodes.items():
            if node_id == failed_node_id:
                continue
                
            similarity = self._calculate_node_similarity(failed_node, node)
            if similarity > 0.5:  # Threshold for connection
                # Determine edge type based on node types
                if node.node_type == ReasoningNodeType.CONTRADICTION:
                    edge_type = ReasoningEdgeType.CONTRADICTION
                elif node.node_type == ReasoningNodeType.QUESTION:
                    edge_type = ReasoningEdgeType.QUESTION
                else:
                    edge_type = ReasoningEdgeType.REVISION
                
                # Create bidirectional edges
                edge1 = ReasoningEdge(
                    source_id=node_id,
                    target_id=failed_node_id,
                    edge_type=edge_type,
                    weight=similarity,
                    metadata={"similarity": similarity},
                    created_by="system"
                )
                scaffold.add_edge(edge1)
                
                edge2 = ReasoningEdge(
                    source_id=failed_node_id,
                    target_id=node_id,
                    edge_type=edge_type,
                    weight=similarity,
                    metadata={"similarity": similarity},
                    created_by="system"
                )
                scaffold.add_edge(edge2)
                
                connected = True
        
        return connected
    
    def create_recursive_loop(self, scaffold: RecursiveScaffold, 
                             source_id: str, target_id: str, 
                             content: str, created_by: str) -> str:
        """Create a recursive loop between two nodes"""
        # Create the recursive connection node
        loop_id = f"loop_{len(scaffold.loop_history)}"
        
        loop_node = ReasoningNode(
            id=loop_id,
            content=content,
            node_type=ReasoningNodeType.ECHO,
            metadata={"source": source_id, "target": target_id},
            created_by=created_by,
            confidence=0.8
        )
        scaffold.add_node(loop_node)
        
        # Connect source to loop node
        source_edge = ReasoningEdge(
            source_id=source_id,
            target_id=loop_id,
            edge_type=ReasoningEdgeType.RECURSION,
            weight=1.0,
            metadata={},
            created_by=created_by
        )
        scaffold.add_edge(source_edge)
        
        # Connect loop node to target
        target_edge = ReasoningEdge(
            source_id=loop_id,
            target_id=target_id,
            edge_type=ReasoningEdgeType.RECURSION,
            weight=1.0,
            metadata={},
            created_by=created_by
        )
        scaffold.add_edge(target_edge)
        
        return loop_id


# ======================================================
# Loop Detection and Management
# ======================================================

class LoopDetector:
    """
    Detects and manages reasoning loops in recursive scaffolds.
    
    The detector identifies when reasoning begins to exhibit recursive
    patterns, distinguishes productive from non-productive loops,
    and helps manage loop closure to support effective reasoning.
    """
    
    def __init__(self, recursion_threshold: float = 0.7, max_depth: int = 5):
        self.recursion_threshold = recursion_threshold
        self.max_depth = max_depth
        self.pattern_memory = {}  # Memory of previously seen patterns
    
    def detect_loops(self, scaffold: RecursiveScaffold) -> List[Dict[str, Any]]:
        """Detect reasoning loops in the scaffold"""
        # Find graph cycles
        cycles = scaffold.detect_loops()
        
        # Analyze each cycle
        loop_data = []
        for cycle in cycles:
            if len(cycle) > 1:  # Ignore self-loops
                loop_info = self._analyze_loop(scaffold, cycle)
                loop_data.append(loop_info)
        
        return loop_data
    
    def _analyze_loop(self, scaffold: RecursiveScaffold, cycle: List[str]) -> Dict[str, Any]:
        """Analyze a detected reasoning loop"""
        nodes = [scaffold.nodes[node_id] for node_id in cycle]
        
        # Calculate basic loop properties
        length = len(cycle)
        complexity = self._calculate_loop_complexity(scaffold, cycle)
        productivity = self._assess_loop_productivity(scaffold, cycle)
        
        # Determine loop type
        loop_type = self._classify_loop_type(scaffold, cycle)
        
        return {
            "cycle": cycle,
            "length": length,
            "complexity": complexity,
            "productivity": productivity,
            "type": loop_type,
            "nodes": [node.content for node in nodes]
        }
    
    def _calculate_loop_complexity(self, scaffold: RecursiveScaffold, cycle: List[str]) -> float:
        """Calculate the complexity of a reasoning loop"""
        # Simple complexity measure based on edge types and node diversity
        edge_type_diversity = len(set(scaffold.graph[s][t]["edge_type"] for s, t in zip(cycle, cycle[1:] + [cycle[0]])))
        node_type_diversity = len(set(scaffold.nodes[n].node_type for n in cycle))
        
        # Normalize to 0-1 range
        return (edge_type_diversity + node_type_diversity) / (2 * max(1, len(cycle)))
    
    def _assess_loop_productivity(self, scaffold: RecursiveScaffold, cycle: List[str]) -> float:
        """Assess the productivity of a reasoning loop"""
        # This would use more sophisticated measures in practice
        # For now, we'll use a simple heuristic based on node confidence
        
        # Calculate average node confidence
        avg_confidence = sum(scaffold.nodes[n].confidence for n in cycle) / len(cycle)
        
        # Check if the loop leads to a conclusion or resolution
        leads_to_conclusion = any(
            scaffold.nodes[t].node_type in [ReasoningNodeType.CONCLUSION, ReasoningNodeType.RESOLUTION]
            for s, t in scaffold.graph.out_edges(cycle[-1])
        )
        
        # Boost productivity score if loop leads to conclusion
        productivity = avg_confidence
        if leads_to_conclusion:
            productivity = min(1.0, productivity * 1.5)
            
        return productivity
    
    def _classify_loop_type(self, scaffold: RecursiveScaffold, cycle: List[str]) -> str:
        """Classify the type of reasoning loop"""
        # Check edge types in the loop
        edge_types = [scaffold.graph[s][t]["edge_type"] for s, t in zip(cycle, cycle[1:] + [cycle[0]])]
        
        # Check node types in the loop
        node_types = [scaffold.nodes[n].node_type.value for n in cycle]
        
        # Classification logic
        if ReasoningEdgeType.REFLECTION.value in edge_types:
            return "reflective"
        elif ReasoningEdgeType.RECURSION.value in edge_types:
            return "recursive"
        elif ReasoningEdgeType.CONTRADICTION.value in edge_types:
            return "contradictory"
        elif ReasoningNodeType.QUESTION.value in node_types:
            return "inquiry"
        else:
            return "standard"
    
    def suggest_loop_closure(self, scaffold: RecursiveScaffold, cycle: List[str]) -> Dict[str, Any]:
        """Suggest how to close a reasoning loop productively"""
        loop_type = self._classify_loop_type(scaffold, cycle)
        
        if loop_type == "contradictory":
            return self._suggest_contradiction_resolution(scaffold, cycle)
        elif loop_type == "inquiry":
            return self._suggest_inquiry_resolution(scaffold, cycle)
        elif loop_type == "reflective":
            return self._suggest_reflection_integration(scaffold, cycle)
        else:
            return self._suggest_general_closure(scaffold, cycle)
    
    def _suggest_contradiction_resolution(self, scaffold: RecursiveScaffold, cycle: List[str]) -> Dict[str, Any]:
        """Suggest how to resolve a contradictory loop"""
        # Find contradiction edges
        contradiction_edges = [(s, t) for s, t in zip(cycle, cycle[1:] + [cycle[0]])
                              if scaffold.graph[s][t]["edge_type"] == ReasoningEdgeType.CONTRADICTION.value]
        
        # Extract the contradicting nodes
        contradiction_nodes = []
        for s, t in contradiction_edges:
            contradiction_nodes.extend([s, t])
        
        return {
            "suggestion_type": "resolve_contradiction",
            "contradiction_nodes": contradiction_nodes,
            "resolution_approach": "Create a resolution node that reconciles or disambiguates the contradicting elements",
            "template": "While {node1} and {node2} appear to contradict, they can be reconciled by considering {suggestion}"
        }
    
    def _suggest_inquiry_resolution(self, scaffold: RecursiveScaffold, cycle: List[str]) -> Dict[str, Any]:
        """Suggest how to resolve an inquiry loop"""
        # Find question nodes
        question_nodes = [n for n in cycle if scaffold.nodes[n].node_type == ReasoningNodeType.QUESTION]
        
        return {
            "suggestion_type": "answer_inquiry",
            "question_nodes": question_nodes,
            "resolution_approach": "Provide definitive answers to the questions in the loop",
            "template": "To resolve this inquiry loop, address the question in {question_node} by determining {suggestion}"
        }
    
    def _suggest_reflection_integration(self, scaffold: RecursiveScaffold, cycle: List[str]) -> Dict[str, Any]:
        """Suggest how to integrate reflection in a loop"""
        # Find reflection nodes
        reflection_nodes = [n for n in cycle if scaffold.nodes[n].node_type == ReasoningNodeType.REFLECTION]
        
        return {
            "suggestion_type": "integrate_reflection",
            "reflection_nodes": reflection_nodes,
            "resolution_approach": "Incorporate the insights from reflection into the main reasoning path",
            "template": "The reflection in {reflection_node} suggests that we should modify our approach by {suggestion}"
        }
    
    def _suggest_general_closure(self, scaffold: RecursiveScaffold, cycle: List[str]) -> Dict[str, Any]:
        """Suggest general closure for a reasoning loop"""
        return {
            "suggestion_type": "general_closure",
            "cycle": cycle,
            "resolution_approach": "Summarize the insights from the loop and advance to a conclusion",
            "template": "This reasoning loop reveals {insight}, which leads us to conclude that {conclusion}"
        }


# ======================================================
# Symbolic Residue Tracker
# ======================================================

class SymbolicResidueTracker:
    """
    Tracks and analyzes symbolic residue from reasoning attempts.
    
    The tracker identifies patterns in failed reasoning attempts,
    extracts useful information from failures, and helps transform
    residue into scaffolding for future reasoning.
    """
    
    def __init__(self):
        self.residue_patterns = {}  # Library of known residue patterns
        self.pattern_counts = {}  # Counts of observed patterns
    
    def register_residue(self, residue: SymbolicResidue) -> None:
        """Register a new symbolic residue instance"""
        signature = residue.pattern_signature
        
        # Update pattern counts
        if signature not in self.pattern_counts:
            self.pattern_counts[signature] = 0
        self.pattern_counts[signature] += 1
        
        # Update pattern library
        if signature not in self.residue_patterns:
            self.residue_patterns[signature] = {
                "signature": signature,
                "examples": [],
                "common_context": {},
                "success_transforms": []
            }
        
        # Add this example (limiting to prevent memory bloat)
        if len(self.residue_patterns[signature]["examples"]) < 10:
            self.residue_patterns[signature]["examples"].append(residue)
        
        # Update common context elements
        self._update_common_context(signature, residue.failure_context)
    
    def _update_common_context(self, signature: str, context: Dict[str, Any]) -> None:
        """Update the common context elements for a pattern"""
        if not self.residue_patterns[signature]["common_context"]:
            # First example, just copy the context
            self.residue_patterns[signature]["common_context"] = context.copy()
        else:
            # Update to keep only common elements
            common_context = {}
            for key, value in self.residue_patterns[signature]["common_context"].items():
                if key in context and context[key] == value:
                    common_context[key] = value
            self.residue_patterns[signature]["common_context"] = common_context
    
    def register_successful_transform(self, residue: SymbolicResidue, 
                                   transform_description: str,
                                   transformed_content: str) -> None:
        """Register a successful transformation of residue into useful reasoning"""
        signature = residue.pattern_signature
        
        if signature in self.residue_patterns:
            transform = {
                "original": residue.content,
                "transform_description": transform_description,
                "transformed": transformed_content
            }
            self.residue_patterns[signature]["success_transforms"].append(transform)
    
    def suggest_transformation(self, residue: SymbolicResidue) -> Optional[Dict[str, Any]]:
        """Suggest a transformation for a given residue based on past successes"""
        signature = residue.pattern_signature
        
        if signature not in self.residue_patterns:
            return None
            
        transforms = self.residue_patterns[signature]["success_transforms"]
        if not transforms:
            return None
            
        # Find the most similar previous example
        best_match = None
        best_similarity = -1
        
        for transform in transforms:
            similarity = self._calculate_text_similarity(residue.content, transform["original"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = transform
        
        if best_similarity > 0.5 and best_match:  # Threshold for suggestion
            return {
                "original_residue": residue.content,
                "similar_example": best_match["original"],
                "transform_description": best_match["transform_description"],
                "transformed_example": best_match["transformed"],
                "similarity": best_similarity
            }
        
        return None
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        # This would use more sophisticated measures in practice
        # Simple implementation for demonstration
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        return len(intersection) / (len(words1) + len(words2) - len(intersection))  # Jaccard similarity
    
    def get_pattern_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics about observed residue patterns"""
        stats = {}
        
        for signature, count in self.pattern_counts.items():
            pattern_data = self.residue_patterns.get(signature, {})
            success_rate = 0
            if signature in self.residue_patterns:
                transforms = len(self.residue_patterns[signature]["success_transforms"])
                success_rate = transforms / max(1, count)
                
            stats[signature] = {
                "count": count,
                "success_transforms": len(pattern_data.get("success_transforms", [])),
                "success_rate": success_rate,
                "examples": len(pattern_data.get("examples", [])),
                "common_context_keys": list(pattern_data.get("common_context", {}).keys())
            }
        
        return stats
    
    def extract_recurring_motifs(self) -> Dict[str, List[str]]:
        """Extract recurring motifs from residue patterns"""
        motifs = {}
        
        for signature, pattern in self.residue_patterns.items():
            examples = pattern.get("examples", [])
            if len(examples) < 3:  # Need at least 3 examples to identify motifs
                continue
                
            # Extract common n-grams across examples
            common_ngrams = self._extract_common_ngrams([ex.content for ex in examples])
            if common_ngrams:
                motifs[signature] = common_ngrams
        
        return motifs
    
    def _extract_common_ngrams(self, texts: List[str], min_length: int = 3, max_length: int = 6) -> List[str]:
        """Extract common n-grams from a list of texts"""
        # This would use more sophisticated analysis in practice
        # Simple implementation for demonstration
        
        all_ngrams = {}
        
        for text in texts:
            words = text.lower().split()
            text_ngrams = set()
            
            for n in range(min_length, min(max_length + 1, len(words) + 1)):
                for i in range(len(words) - n + 1):
                    ngram = " ".join(words[i:i+n])
                    text_ngrams.add(ngram)
            
            for ngram in text_ngrams:
                all_ngrams[ngram] = all_ngrams.get(ngram, 0) + 1
        
        # Find ngrams that appear in at least half the texts
        common = [ngram for ngram, count in all_ngrams.items() if count >= len(texts) / 2]
        
        # Sort by frequency and length
        return sorted(common, key=lambda x: (all_ngrams[x], len(x)), reverse=True)[:10]


# ======================================================
# Recursion Graph Analysis
# ======================================================

class RecursionGraphAnalyzer:
    """
    Analyzes recursion patterns in reasoning scaffolds.
    
    The analyzer represents recursive reasoning structures as graphs,
    computes graph properties, and identifies patterns that support
    effective recursive reasoning.
    """
    
    def __init__(self):
        self.graph_metrics = {}  # Cache for computed metrics
    
    def analyze_scaffold(self, scaffold: RecursiveScaffold) -> Dict[str, Any]:
        """Analyze the recursive properties of a scaffold"""
        graph = scaffold.graph
        
        # Compute basic graph metrics
        metrics = {
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "density": nx.density(graph),
            "average_clustering": nx.average_clustering(graph),
            "strongly_connected_components": list(nx.strongly_connected_components(graph)),
            "recursive_paths": scaffold.get_recursion_paths(),
            "recursion_depth": scaffold.calculate_recursion_depth()
        }
        
        # Compute centrality measures for all nodes
        try:
            metrics["degree_centrality"] = nx.degree_centrality(graph)
            metrics["eigenvector_centrality"] = nx.eigenvector_centrality(graph)
            metrics["betweenness_centrality"] = nx.betweenness_centrality(graph)
        except:
            # Some centrality measures may fail on certain graph structures
            logger.warning("Failed to compute some centrality measures")
        
        # Cache the metrics
        self.graph_metrics[scaffold.name] = metrics
        
        return metrics
    
    def identify_key_nodes(self, scaffold: RecursiveScaffold, top_n: int = 5) -> Dict[str, List[str]]:
        """Identify key nodes in the scaffold based on centrality measures"""
        # Make sure metrics are computed
        if scaffold.name not in self.graph_metrics:
            self.analyze_scaffold(scaffold)
            
        metrics = self.graph_metrics[scaffold.name]
        
        # Identify top nodes by different centrality measures
        key_nodes = {}
        
        for metric_name in ["degree_centrality", "eigenvector_centrality", "betweenness_centrality"]:
            if metric_name in metrics:
                centrality = metrics[metric_name]
                top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
                key_nodes[metric_name] = [node_id for node_id, _ in top_nodes]
        
        return key_nodes
    
    def identify_feedback_loops(self, scaffold: RecursiveScaffold) -> List[List[str]]:
        """Identify feedback loops in the reasoning scaffold"""
        # A feedback loop is a cycle that contains at least one reflection edge
        cycles = list(nx.simple_cycles(scaffold.graph))
        
        feedback_loops = []
        for cycle in cycles:
            # Check if cycle contains a reflection edge
            has_reflection = False
            for i in range(len(cycle)):
                s, t = cycle[i], cycle[(i+1) % len(cycle)]
                if scaffold.graph[s][t].get("edge_type") == ReasoningEdgeType.REFLECTION.value:
                    has_reflection = True
                    break
                    
            if has_reflection:
                feedback_loops.append(cycle)
        
        return feedback_loops
    
    def compute_scaffold_complexity(self, scaffold: RecursiveScaffold) -> float:
        """Compute a complexity score for the reasoning scaffold"""
        # Make sure metrics are computed
        if scaffold.name not in self.graph_metrics:
            self.analyze_scaffold(scaffold)
            
        metrics = self.graph_metrics[scaffold.name]
        
        # Compute complexity based on multiple factors
        # Higher values indicate more complex scaffolds
        
        # Factor 1: Graph size and connectivity
        size_factor = np.log1p(metrics["node_count"] * metrics["density"])
        
        # Factor 2: Recursion depth and paths
        recursion_factor = metrics["recursion_depth"] * np.log1p(len(metrics["recursive_paths"]))
        
        # Factor 3: Node type diversity
        node_types = set(data["node_type"] for _, data in scaffold.graph.nodes(data=True))
        diversity_factor = len(node_types) / len(ReasoningNodeType)
        
        # Factor 4: Edge type diversity
        edge_types = set(data["edge_type"] for _, _, data in scaffold.graph.edges(data=True))
        edge_diversity_factor = len(edge_types) / len(ReasoningEdgeType)
        
        # Combine factors into overall complexity score
        complexity = size_factor + recursion_factor + diversity_factor + edge_diversity_factor
        
        return complexity
    
    def compare_scaffolds(self, scaffold1: RecursiveScaffold, scaffold2: RecursiveScaffold) -> Dict[str, Any]:
        """Compare two reasoning scaffolds"""
        # Make sure metrics are computed
        if scaffold1.name not in self.graph_metrics:
            self.analyze_scaffold(scaffold1)
        if scaffold2.name not in self.graph_metrics:
            self.analyze_scaffold(scaffold2)
            
        metrics1 = self.graph_metrics[scaffold1.name]
        metrics2 = self.graph_metrics[scaffold2.name]
        
        # Compare basic metrics
        comparison = {
            "node_count_diff": metrics2["node_count"] - metrics1["node_count"],
            "edge_count_diff": metrics2["edge_count"] - metrics1["edge_count"],
            "density_ratio": metrics2["density"] / max(0.001, metrics1["density"]),
            "clustering_ratio": metrics2["average_clustering"] / max(0.001, metrics1["average_clustering"]),
            "recursion_depth_diff": metrics2["recursion_depth"] - metrics1["recursion_depth"]
        }
        
        # Compare complexity
        complexity1 = self.compute_scaffold_complexity(scaffold1)
        complexity2 = self.compute_scaffold_complexity(scaffold2)
        comparison["complexity_ratio"] = complexity2 / max(0.001, complexity1)
        
        return comparison
    
    def suggest_scaffold_improvements(self, scaffold: RecursiveScaffold) -> List[Dict[str, Any]]:
        """Suggest improvements to a reasoning scaffold"""
        # Make sure metrics are computed
        if scaffold.name not in self.graph_metrics:
            self.analyze_scaffold(scaffold)
            
        suggestions = []
        
        # Suggestion 1: Add reflection nodes for complex subgraphs
        suggestion = self._suggest_reflection_nodes(scaffold)
        if suggestion:
            suggestions.append(suggestion)
        
        # Suggestion 2: Close potential recursive loops
        suggestion = self._suggest_loop_closures(scaffold)
        if suggestion:
            suggestions.append(suggestion)
        
        # Suggestion 3: Add resolution nodes for contradictions
        suggestion = self._suggest_contradiction_resolutions(scaffold)
        if suggestion:
            suggestions.append(suggestion)
        
        # Suggestion 4: Connect isolated components
        suggestion = self._suggest_component_connections(scaffold)
        if suggestion:
            suggestions.append(suggestion)
        
        return suggestions
    
    def _suggest_reflection_nodes(self, scaffold: RecursiveScaffold) -> Optional[Dict[str, Any]]:
        """Suggest adding reflection nodes for complex subgraphs"""
        # Identify complex subgraphs with no reflection nodes
        metrics = self.graph_metrics[scaffold.name]
        
        # Get strongly connected components with at least 3 nodes
        large_components = [comp for comp in metrics["strongly_connected_components"] if len(comp) >= 3]
        
        if not large_components:
            return None
            
        # Check if components have reflection nodes
        components_without_reflection = []
        for comp in large_components:
            has_reflection = False
            for node_id in comp:
                if scaffold.nodes[node_id].node_type == ReasoningNodeType.REFLECTION:
                    has_reflection = True
                    break
                    
            if not has_reflection:
                components_without_reflection.append(comp)
        
        if not components_without_reflection:
            return None
            
        return {
            "suggestion_type": "add_reflection",
            "components": components_without_reflection,
            "description": "Add reflection nodes to complex subgraphs to enhance meta-cognition",
            "template": "Consider reflecting on the reasoning in nodes {nodes} to identify patterns and improve understanding"
        }
    
    def _suggest_loop_closures(self, scaffold: RecursiveScaffold) -> Optional[Dict[str, Any]]:
        """Suggest closing potential recursive loops"""
        # Look for almost-cycles - paths that could form cycles with one extra edge
        potential_loops = []
        
        for source in scaffold.graph.nodes():
            for target in scaffold.graph.nodes():
                if source != target and not scaffold.graph.has_edge(source, target):
                    # Check if there's a path from target to source
                    try:
                        path = nx.shortest_path(scaffold.graph, target, source)
                        if 2 <= len(path) <= 5:  # Reasonable loop size
                            potential_loops.append((source, target, path))
                    except nx.NetworkXNoPath:
                        pass
        
        if not potential_loops:
            return None
            
        # Sort by path length (prefer shorter loops)
        potential_loops.sort(key=lambda x: len(x[2]))
        
        # Take top 3 suggestions
        top_suggestions = potential_loops[:3]
        
        return {
            "suggestion_type": "close_loops",
            "potential_loops": [{"source": s, "target": t, "path": p} for s, t, p in top_suggestions],
            "description": "Close potential reasoning loops to enable recursive reasoning",
            "template": "Consider connecting node {source} back to node {target} to form a recursive reasoning loop"
        }
    
    def _suggest_contradiction_resolutions(self, scaffold: RecursiveScaffold) -> Optional[Dict[str, Any]]:
        """Suggest resolving contradictions in the scaffold"""
        # Find contradiction edges
        contradiction_edges = [(s, t) for s, t, d in scaffold.graph.edges(data=True) 
                             if d.get("edge_type") == ReasoningEdgeType.CONTRADICTION.value]
        
        if not contradiction_edges:
            return None
            
        # Find contradiction edges without resolution
        unresolved = []
        for s, t in contradiction_edges:
            has_resolution = False
            for _, r, d in scaffold.graph.out_edges([s, t], data=True):
                if d.get("edge_type") == ReasoningEdgeType.RESOLUTION.value:
                    has_resolution = True
                    break
                    
            if not has_resolution:
                unresolved.append((s, t))
