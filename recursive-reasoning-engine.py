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
