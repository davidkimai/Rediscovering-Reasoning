"""
Pattern Resonance Mapping
=========================

This module implements the core pattern resonance mapping capabilities for the Rediscovering Reasoning 
framework. It focuses on identifying, encoding, and leveraging recurring patterns in reasoning 
attempts to build more robust recursive reasoning scaffolds.

Key components:
1. Resonance Detector - Identifies recurring patterns across reasoning attempts
2. Symbolic Encoder - Encodes recurring patterns in symbolic form
3. Failure Reframing - Transforms reasoning failures into opportunities
4. Emergence Tracker - Monitors the emergence of novel patterns

The pattern resonance approach enables the framework to identify and cultivate
emergent reasoning structures that transcend linear proof steps.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import re
from collections import defaultdict, Counter
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import the core engine components
try:
    from engine.scaffold import (
        RecursiveScaffold, ReasoningNode, ReasoningEdge, 
        ReasoningNodeType, ReasoningEdgeType, SymbolicResidue
    )
except ImportError:
    logger.warning("Unable to import core engine components. Using local definitions.")
    
    # Simple placeholder enums if imports fail
    class ReasoningNodeType(Enum):
        PREMISE = "premise"
        INFERENCE = "inference"
        CONCLUSION = "conclusion"
        REFLECTION = "reflection"
        QUESTION = "question"
        CONTRADICTION = "contradiction"
        RESOLUTION = "resolution"
        RESIDUE = "residue"
        ECHO = "echo"
        
    class ReasoningEdgeType(Enum):
        DEDUCTION = "deduction"
        REFLECTION = "reflection"
        CONTRADICTION = "contradiction"
        RESOLUTION = "resolution"
        RECURSION = "recursion"
    
    # Simple placeholder classes
    @dataclass
    class SymbolicResidue:
        content: str
        pattern_signature: str
        confidence: float
        
    @dataclass
    class ReasoningNode:
        id: str
        content: str
        node_type: ReasoningNodeType
        
    @dataclass
    class ReasoningEdge:
        source_id: str
        target_id: str
        edge_type: ReasoningEdgeType
        
    class RecursiveScaffold:
        def __init__(self):
            self.nodes = {}
            self.graph = nx.DiGraph()


# ======================================================
# Data Structures for Pattern Resonance
# ======================================================

@dataclass
class ResonancePattern:
    """Represents a resonance pattern identified across reasoning attempts"""
    id: str                            # Unique identifier for the pattern
    name: str                          # Human-readable name
    description: str                   # Description of the pattern
    signature: str                     # Unique signature for pattern matching
    frequency: int = 0                 # Number of times pattern has been observed
    instances: List[str] = None        # List of text instances of this pattern
    symbolic_form: str = None          # Symbolic representation
    embeddings: List[List[float]] = None  # Vector embeddings of instances
    metadata: Dict[str, Any] = None    # Additional metadata
    
    def __post_init__(self):
        if self.instances is None:
            self.instances = []
        if self.embeddings is None:
            self.embeddings = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SymbolicEncoding:
    """Represents a symbolic encoding of a reasoning pattern"""
    pattern_id: str                    # ID of the encoded pattern
    symbol: str                        # Symbolic representation
    encoding_rule: str                 # Rule used for encoding
    compression_ratio: float           # Ratio of original to encoded size
    reversible: bool                   # Whether encoding is reversible
    metadata: Dict[str, Any] = None    # Additional metadata
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ReframedFailure:
    """Represents a reframed reasoning failure"""
    original_content: str              # Original failed reasoning
    reframed_content: str              # Reframed version
    transformation: str                # Description of the transformation
    pattern_id: str                    # ID of the detected pattern
    effectiveness: float = 0.0         # Effectiveness rating (0.0-1.0)
    metadata: Dict[str, Any] = None    # Additional metadata
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class EmergentPattern:
    """Represents an emergent pattern tracked over time"""
    id: str                            # Unique identifier
    description: str                   # Description of the pattern
    first_observed: str                # Timestamp of first observation
    evolution: List[Dict[str, Any]]    # History of pattern evolution
    stability: float                   # Measure of pattern stability (0.0-1.0)
    metadata: Dict[str, Any] = None    # Additional metadata
    
    def __post_init__(self):
        if self.evolution is None:
            self.evolution = []
        if self.metadata is None:
            self.metadata = {}


# ======================================================
# Resonance Detector
# ======================================================

class ResonanceDetector:
    """
    Identifies recurring patterns across reasoning attempts.
    
    The detector finds common structures, motifs, and approaches
    that appear in multiple reasoning attempts, measuring their
    resonance strength and tracking their evolution.
    """
    
    def __init__(self, embedding_model=None, similarity_threshold=0.7):
        self.patterns = {}  # Dictionary of identified patterns
        self.embedding_model = embedding_model  # Model for embedding text
        self.similarity_threshold = similarity_threshold
        self.pattern_counter = 0  # Counter for generating pattern IDs
    
    def compute_embedding(self, text: str) -> List[float]:
        """Compute embedding for text using the embedding model"""
        if self.embedding_model:
            return self.embedding_model(text)
        else:
            # Simple fallback: count character frequencies as a crude embedding
            chars = set(text.lower())
            embedding = [text.lower().count(c) / len(text) for c in sorted(chars)]
            # Normalize
            norm = np.sqrt(sum(x*x for x in embedding))
            return [x/norm for x in embedding] if norm > 0 else embedding
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute similarity between two embeddings"""
        # Cosine similarity
        if not embedding1 or not embedding2:
            return 0.0
        
        dot_product = sum(a*b for a, b in zip(embedding1, embedding2))
        norm1 = np.sqrt(sum(x*x for x in embedding1))
        norm2 = np.sqrt(sum(x*x for x in embedding2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def detect_pattern(self, text: str) -> Optional[str]:
        """
        Detect if text matches an existing pattern
        
        Returns pattern ID if match found, None otherwise
        """
        if not text.strip():
            return None
            
        # Compute embedding for the text
        embedding = self.compute_embedding(text)
        
        # Check for similarity with existing patterns
        best_match = None
        best_similarity = -1
        
        for pattern_id, pattern in self.patterns.items():
            # Check each instance of the pattern
            for inst_embedding in pattern.embeddings:
                similarity = self.compute_similarity(embedding, inst_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = pattern_id
        
        if best_similarity >= self.similarity_threshold:
            return best_match
        
        return None
    
    def register_pattern(self, text: str, name: str = None, description: str = None) -> str:
        """
        Register a new pattern or add an instance to an existing pattern
        
        Returns the pattern ID
        """
        # Check if text matches an existing pattern
        pattern_id = self.detect_pattern(text)
        
        if pattern_id:
            # Add this instance to the existing pattern
            pattern = self.patterns[pattern_id]
            pattern.instances.append(text)
            pattern.embeddings.append(self.compute_embedding(text))
            pattern.frequency += 1
            return pattern_id
        
        # Create a new pattern
        self.pattern_counter += 1
        new_id = f"pattern_{self.pattern_counter}"
        
        if not name:
            name = f"Pattern {self.pattern_counter}"
        if not description:
            description = f"Automatically detected pattern from text: {text[:50]}..."
            
        # Generate a signature from the text
        signature = self._generate_signature(text)
        
        # Create the pattern
        pattern = ResonancePattern(
            id=new_id,
            name=name,
            description=description,
            signature=signature,
            frequency=1,
            instances=[text],
            embeddings=[self.compute_embedding(text)]
        )
        
        self.patterns[new_id] = pattern
        return new_id
    
    def _generate_signature(self, text: str) -> str:
        """Generate a unique signature for a pattern based on text"""
        # Extract key phrases and structural elements
        sentences = text.split('.')
        words = text.split()
        
        # Extract key markers
        markers = []
        
        # Look for specific phrases
        phrase_patterns = [
            r"if .*? then",
            r"because .*?(?=,|\.|$)",
            r"therefore .*?(?=,|\.|$)",
            r"assume .*?(?=,|\.|$)",
            r"let .*?(?=,|\.|$)",
            r"contradiction",
            r"however",
            r"proves"
        ]
        
        for pattern in phrase_patterns:
            matches = re.findall(pattern, text.lower())
            markers.extend(matches)
        
        # Get common n-grams if we have enough text
        if len(words) > 5:
            # Extract 3-grams
            ngrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            # Take most frequent
            if ngrams:
                counter = Counter(ngrams)
                markers.extend([n for n, _ in counter.most_common(3)])
        
        # Use sentence structure if available
        if len(sentences) > 1:
            # Get first and last sentence fragments
            markers.append(sentences[0][:20] if len(sentences[0]) > 20 else sentences[0])
            markers.append(sentences[-1][:20] if len(sentences[-1]) > 20 else sentences[-1])
        
        # Combine markers into a signature
        signature = '|'.join(markers)
        if not signature:
            # Fallback to hash of text if no markers
            signature = f"hash_{hash(text) % 10000:04d}"
            
        return signature
    
    def measure_resonance(self, pattern_id: str) -> Dict[str, Any]:
        """Measure the resonance strength of a pattern"""
        pattern = self.patterns.get(pattern_id)
        if not pattern:
            return {"resonance": 0, "consistency": 0, "evolution": 0}
            
        # Frequency component
        frequency_score = min(1.0, pattern.frequency / 10)  # Cap at 10 instances
        
        # Consistency component - how similar are instances?
        consistency_score = 0
        if len(pattern.embeddings) > 1:
            similarities = []
            for i in range(len(pattern.embeddings)):
                for j in range(i+1, len(pattern.embeddings)):
                    similarities.append(
                        self.compute_similarity(pattern.embeddings[i], pattern.embeddings[j])
                    )
            consistency_score = sum(similarities) / len(similarities) if similarities else 0
        
        # Combine into overall resonance score
        resonance_score = (frequency_score + consistency_score) / 2
        
        return {
            "resonance": resonance_score,
            "frequency": frequency_score,
            "consistency": consistency_score,
            "instances": pattern.frequency
        }
    
    def detect_patterns_in_scaffold(self, scaffold: RecursiveScaffold) -> Dict[str, List[str]]:
        """
        Detect patterns in a reasoning scaffold
        
        Returns a mapping of pattern IDs to lists of node IDs
        """
        pattern_instances = defaultdict(list)
        
        # Process each node in the scaffold
        for node_id, node in scaffold.nodes.items():
            pattern_id = self.detect_pattern(node.content)
            if pattern_id:
                pattern_instances[pattern_id].append(node_id)
            else:
                # Register as a new pattern
                new_pattern_id = self.register_pattern(
                    node.content,
                    f"Pattern from {node_id}",
                    f"Pattern extracted from node {node_id} of type {node.node_type}"
                )
                pattern_instances[new_pattern_id].append(node_id)
        
        return dict(pattern_instances)
    
    def find_cross_scaffold_patterns(self, scaffolds: List[RecursiveScaffold]) -> Dict[str, Dict[str, List[str]]]:
        """
        Find patterns that appear across multiple scaffolds
        
        Returns a mapping of pattern IDs to scaffolds and node IDs
        """
        cross_patterns = {}
        
        # First, detect patterns in each scaffold
        scaffold_patterns = {}
        for scaffold in scaffolds:
            scaffold_patterns[scaffold.name] = self.detect_patterns_in_scaffold(scaffold)
        
        # Find patterns that appear in multiple scaffolds
        all_patterns = set()
        for patterns in scaffold_patterns.values():
            all_patterns.update(patterns.keys())
        
        for pattern_id in all_patterns:
            # Count scaffolds containing this pattern
            scaffold_count = sum(1 for s_patterns in scaffold_patterns.values() 
                                if pattern_id in s_patterns)
            
            if scaffold_count > 1:  # Pattern appears in multiple scaffolds
                cross_patterns[pattern_id] = {
                    "scaffolds": scaffold_count,
                    "instances": {
                        scaffold.name: scaffold_patterns[scaffold.name][pattern_id]
                        for scaffold in scaffolds
                        if pattern_id in scaffold_patterns[scaffold.name]
                    },
                    "pattern": self.patterns[pattern_id]
                }
        
        return cross_patterns
    
    def extract_recurring_motifs(self) -> Dict[str, Any]:
        """Extract recurring motifs from observed patterns"""
        motifs = {}
        
        # Group patterns by similarity
        pattern_clusters = self._cluster_patterns()
        
        # Extract motifs from each cluster
        for cluster_id, pattern_ids in pattern_clusters.items():
            if len(pattern_ids) < 2:  # Need at least 2 patterns for a motif
                continue
                
            # Collect all instances from patterns in this cluster
            instances = []
            for pattern_id in pattern_ids:
                instances.extend(self.patterns[pattern_id].instances)
            
            # Extract common n-grams and phrases
            common_ngrams = self._extract_common_ngrams(instances)
            key_phrases = self._extract_key_phrases(instances)
            
            # Structure as a motif
            motifs[f"motif_{cluster_id}"] = {
                "patterns": pattern_ids,
                "common_ngrams": common_ngrams,
                "key_phrases": key_phrases,
                "instance_count": len(instances)
            }
        
        return motifs
    
    def _cluster_patterns(self) -> Dict[str, List[str]]:
        """Cluster patterns based on similarity"""
        # Build similarity matrix
        pattern_ids = list(self.patterns.keys())
        if not pattern_ids:
            return {}
            
        n = len(pattern_ids)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                # Compare patterns by average embedding similarity
                p1 = self.patterns[pattern_ids[i]]
                p2 = self.patterns[pattern_ids[j]]
                
                # Skip if either pattern has no embeddings
                if not p1.embeddings or not p2.embeddings:
                    continue
                
                # Compute cross-similarities
                similarities = []
                for emb1 in p1.embeddings:
                    for emb2 in p2.embeddings:
                        similarities.append(self.compute_similarity(emb1, emb2))
                
                avg_similarity = sum(similarities) / len(similarities) if similarities else 0
                similarity_matrix[i, j] = avg_similarity
                similarity_matrix[j, i] = avg_similarity
        
        # Simple threshold-based clustering
        clusters = {}
        cluster_id = 0
        unassigned = set(range(n))
        
        while unassigned:
            # Start a new cluster
            cluster = []
            seed = unassigned.pop()
            cluster.append(seed)
            
            # Find all patterns similar to the seed
            for i in unassigned.copy():
                if similarity_matrix[seed, i] >= self.similarity_threshold:
                    cluster.append(i)
                    unassigned.remove(i)
            
            # Store the cluster
            clusters[f"cluster_{cluster_id}"] = [pattern_ids[i] for i in cluster]
            cluster_id += 1
        
        return clusters
    
    def _extract_common_ngrams(self, texts: List[str], min_length: int = 3, max_length: int = 5) -> List[str]:
        """Extract common n-grams from a list of texts"""
        if not texts:
            return []
            
        # Tokenize texts
        tokenized = [text.lower().split() for text in texts]
        
        # Extract n-grams
        all_ngrams = Counter()
        
        for tokens in tokenized:
            for n in range(min_length, min(max_length + 1, len(tokens) + 1)):
                for i in range(len(tokens) - n + 1):
                    ngram = ' '.join(tokens[i:i+n])
                    all_ngrams[ngram] += 1
        
        # Find common n-grams (appear in at least half of texts)
        threshold = max(2, len(texts) / 2)
        common = [ngram for ngram, count in all_ngrams.items() if count >= threshold]
        
        # Sort by frequency and length
        return sorted(common, key=lambda x: (all_ngrams[x], len(x)), reverse=True)[:10]
    
    def _extract_key_phrases(self, texts: List[str]) -> List[str]:
        """Extract key phrases from texts"""
        phrases = []
        
        # Look for specific markers of reasoning
        markers = [
            (r"if .*? then", "conditional"),
            (r"because .*?(?=,|\.|$)", "explanation"),
            (r"therefore .*?(?=,|\.|$)", "conclusion"),
            (r"assume .*?(?=,|\.|$)", "assumption"),
            (r"suppose .*?(?=,|\.|$)", "supposition"),
            (r"let .*?(?=,|\.|$)", "definition"),
            (r"contradiction", "contradiction"),
            (r"however", "contrast"),
            (r"moreover", "addition"),
            (r"implies", "implication")
        ]
        
        for text in texts:
            for pattern, label in markers:
                matches = re.findall(pattern, text.lower())
                for match in matches:
                    phrases.append((match, label))
        
        # Count and return most common
        phrase_counter = Counter(phrases)
        return [phrase for phrase, _ in phrase_counter.most_common(10)]


# ======================================================
# Symbolic Encoder
# ======================================================

class SymbolicEncoder:
    """
    Encodes recurring patterns in symbolic form.
    
    The encoder creates a vocabulary of recursive reasoning motifs,
    represents them in compact symbolic form, and establishes
    mappings between symbolic representations and reasoning patterns.
    """
    
    def __init__(self):
        self.encodings = {}  # Dictionary of pattern ID to symbolic encoding
        self.symbols = set()  # Set of used symbols
        self.symbol_counter = 0  # Counter for generating symbols
    
    def encode_pattern(self, pattern: ResonancePattern) -> SymbolicEncoding:
        """Encode a pattern in symbolic form"""
        if pattern.id in self.encodings:
            return self.encodings[pattern.id]
            
        # Generate a symbol
        symbol = self._generate_symbol()
        
        # Create encoding rule
        encoding_rule = self._create_encoding_rule(pattern)
        
        # Compute compression ratio
        avg_length = sum(len(inst) for inst in pattern.instances) / len(pattern.instances) if pattern.instances else 0
        compression_ratio = avg_length / len(symbol) if symbol else 0
        
        # Create the encoding
        encoding = SymbolicEncoding(
            pattern_id=pattern.id,
            symbol=symbol,
            encoding_rule=encoding_rule,
            compression_ratio=compression_ratio,
            reversible=True  # Assume encodings are reversible
        )
        
        self.encodings[pattern.id] = encoding
        return encoding
    
    def _generate_symbol(self) -> str:
        """Generate a unique symbol"""
        # Define a set of Unicode symbols for encoding
        symbol_sets = [
            "🔄📊🔍📈📉🧩🔗🔎📋🔖",  # Emoji symbols
            "αβγδεζηθικλμνξοπρστυφχψω",  # Greek letters
            "∀∃∈∉⊂⊃∪∩∧∨¬→↔≡≠≤≥±∞∂∫∑∏√∛∜"  # Mathematical symbols
        ]
        
        # Try to find an unused symbol
        for symbol_set in symbol_sets:
            for symbol in symbol_set:
                if symbol not in self.symbols:
                    self.symbols.add(symbol)
                    return symbol
        
        # If all symbols are used, create a numbered symbol
        self.symbol_counter += 1
        return f"§{self.symbol_counter}"
    
    def _create_encoding_rule(self, pattern: ResonancePattern) -> str:
        """Create a rule for encoding and decoding a pattern"""
        # Extract key elements from pattern
        key_elements = []
        
        # Use the pattern signature as a basis
        signature_parts = pattern.signature.split('|')
        if signature_parts:
            key_elements.extend(signature_parts[:3])  # Limit to first 3 parts
        
        # Get common elements from instances
        if pattern.instances:
            common_ngrams = self._extract_common_elements(pattern.instances)
            key_elements.extend(common_ngrams[:3])  # Limit to first 3 common elements
        
        # Combine into a rule
        if key_elements:
            rule = "Presence of: " + ", ".join(key_elements)
        else:
            rule = "General pattern without specific markers"
            
        return rule
    
    def _extract_common_elements(self, texts: List[str]) -> List[str]:
        """Extract common elements from a list of texts"""
        if not texts:
            return []
            
        # Simple word-based approach
        word_counters = [Counter(text.lower().split()) for text in texts]
        
        # Find words that appear in all texts
        common_words = set(word_counters[0].keys())
        for counter in word_counters[1:]:
            common_words &= set(counter.keys())
        
        # Sort by average frequency
        scored_words = []
        for word in common_words:
            avg_freq = sum(counter[word] for counter in word_counters) / len(word_counters)
            scored_words.append((word, avg_freq))
        
        return [word for word, _ in sorted(scored_words, key=lambda x: x[1], reverse=True)]
    
    def decode_symbol(self, symbol: str) -> Optional[str]:
        """Decode a symbol back to its pattern description"""
        for encoding in self.encodings.values():
            if encoding.symbol == symbol:
                return encoding.pattern_id
        return None
    
    def encode_text(self, text: str, detector: ResonanceDetector) -> str:
        """
        Encode a text by replacing pattern instances with symbols
        
        Uses the detector to identify patterns in the text
        """
        # Find pattern in text
        pattern_id = detector.detect_pattern(text)
        if not pattern_id or pattern_id not in self.encodings:
            return text  # No pattern found or no encoding for pattern
            
        # Get the encoding
        encoding = self.encodings[pattern_id]
        
        # Replace text with symbol
        # This is a simplified approach - in practice we'd need more sophisticated
        # replacement that preserves context and handles partial matches
        return f"{encoding.symbol} [{text[:20]}...]"
    
    def encode_scaffold(self, scaffold: RecursiveScaffold, detector: ResonanceDetector) -> Dict[str, str]:
        """
        Encode nodes in a scaffold using symbolic representations
        
        Returns a dictionary mapping node IDs to encoded content
        """
        encoded_nodes = {}
        
        for node_id, node in scaffold.nodes.items():
            encoded_nodes[node_id] = self.encode_text(node.content, detector)
            
        return encoded_nodes
    
    def create_symbol_vocabulary(self, detector: ResonanceDetector) -> Dict[str, Dict[str, Any]]:
        """
        Create a vocabulary of symbolic representations
        
        Maps symbols to patterns and their meanings
        """
        vocabulary = {}
        
        # Encode all patterns in the detector
        for pattern_id, pattern in detector.patterns.items():
            encoding = self.encode_pattern(pattern)
            
            vocabulary[encoding.symbol] = {
                "pattern_id": pattern_id,
                "pattern_name": pattern.name,
                "description": pattern.description,
                "rule": encoding.encoding_rule,
                "example": pattern.instances[0][:100] + "..." if pattern.instances else "No examples",
                "frequency": pattern.frequency
            }
            
        return vocabulary


# ======================================================
# Failure Reframing
# ======================================================

class FailureReframer:
    """
    Transforms reasoning failures into opportunities for recursion.
    
    The reframer identifies patterns in unsuccessful reasoning attempts,
    reframes failures as opportunities for recursive insight, and
    builds scaffolds that incorporate previously failed attempts.
    """
    
    def __init__(self, detector: ResonanceDetector = None):
        self.reframing_strategies = {}  # Strategies for reframing different failure types
        self.reframings = []  # History of reframed failures
        self.detector = detector or ResonanceDetector()
        
        # Register default reframing strategies
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default reframing strategies for common failure types"""
        self.register_strategy(
            "circular_reasoning",
            "Transform circular reasoning into productive recursive structures",
            self._reframe_circular_reasoning
        )
        
        self.register_strategy(
            "contradiction",
            "Reframe contradictions as opportunities for distinction and clarification",
            self._reframe_contradiction
        )
        
        self.register_strategy(
            "incomplete_proof",
            "Transform incomplete proofs into scaffolds for recursive completion",
            self._reframe_incomplete_proof
        )
        
        self.register_strategy(
            "over_generalization",
            "Reframe over-generalizations as recursive refinement opportunities",
            self._reframe_over_generalization
        )
        
        self.register_strategy(
            "limited_perspective",
            "Transform limited perspectives into recursive viewpoint expansion",
            self._reframe_limited_perspective
        )
    
    def register_strategy(self, strategy_id: str, description: str, 
                        reframe_func: Callable[[str, Dict[str, Any]], str]):
        """Register a new reframing strategy"""
        self.reframing_strategies[strategy_id] = {
            "id": strategy_id,
            "description": description,
            "function": reframe_func,
            "usage_count": 0,
            "success_rate": 0.0
        }
    
    def reframe_failure(self, content: str, context: Dict[str, Any] = None) -> ReframedFailure:
        """
        Reframe a failed reasoning attempt
        
        Args:
            content: The content of the failed reasoning
            context: Additional context about the failure
            
        Returns:
            A ReframedFailure object containing the reframed content
        """
        if context is None:
            context = {}
            
        # Detect failure type
        failure_type = self._detect_failure_type(content, context)
        
        # Select appropriate reframing strategy
        strategy = self.reframing_strategies.get(failure_type)
        if not strategy:
            # Fallback to general reframing
            strategy = self.reframing_strategies.get("incomplete_proof")
            if not strategy:
                return ReframedFailure(
                    original_content=content,
                    reframed_content=content,  # No reframing
                    transformation="No applicable reframing strategy found",
                    pattern_id="unknown",
                    effectiveness=0.0
                )
        
        # Apply the reframing strategy
        try:
            reframed_content = strategy["function"](content, context)
            transformation = strategy["description"]
            strategy["usage_count"] += 1
            
            # Detect pattern in original content
            pattern_id = "unknown"
            if self.detector:
                detected_id = self.detector.detect_pattern(content)
                if detected_id:
                    pattern_id = detected_id
                else:
                    # Register as a new pattern
                    pattern_id = self.detector.register_pattern(
                        content,
                        f"Failure Pattern {failure_type.capitalize()}",
                        f"Automatically detected {failure_type} failure pattern"
                    )
            
            # Create the reframed failure
            reframed = ReframedFailure(
                original_content=content,
                reframed_content=reframed_content,
                transformation=transformation,
                pattern_id=pattern_id,
                effectiveness=0.8,  # Placeholder - would be evaluated in practice
                metadata={"failure_type": failure_type}
            )
            
            # Save to history
            self.reframings.append(reframed)
            return reframed
            
        except Exception as e:
            logger.error(f"Error applying reframing strategy {failure_type}: {e}")
            return ReframedFailure(
                original_content=content,
                reframed_content=content,  # No reframing
                transformation=f"Error in reframing: {str(e)}",
                pattern_id="error",
                effectiveness=0.0
            )
    
    def _detect_failure_type(self, content: str, context: Dict[str, Any]) -> str:
        """Detect the type of failure in the content"""
        # Check for explicit failure type in context
        if "failure_type" in context:
            return context["failure_type"]
        
        # Simple keyword-based detection
        failure_types = {
            "circular_reasoning": ["circular", "loop", "same as", "repeating", "assumes what it proves"],
            "contradiction": ["contradiction", "inconsistent", "cannot be both", "conflict"],
            "incomplete_proof": ["incomplete", "missing", "not enough", "need more", "unfinished"],
            "over_generalization": ["too general", "not always true", "counterexample", "exception"],
            "limited_perspective": ["narrow", "alternative", "also consider", "another view"]
        }
        
        # Check each failure type
        matches = {}
        for ftype, keywords in failure_types.items():
            count = sum(1 for keyword in keywords if keyword in content.lower())
            matches[ftype] = count
        
        # Return the type with the most matches
        if matches:
            return max(matches.items(), key=lambda x: x[1])[0]
            
        # Default to incomplete proof
        return "incomplete_proof"
    
    def _reframe_circular_reasoning(self, content: str, context: Dict[str, Any]) -> str:
        """Reframe circular reasoning into productive recursive structures"""
        # Identify the circular pattern
        sentences = content.split('.')
        circular_parts = []
        
        # Look for repeated phrases or structures
        words = content.lower().split()
        
        # Find repeated n-grams
        repeated = []
        for n in range(3, min(6, len(words) // 2)):  # Look for 3 to 5-word phrases
            for i in range(len(words) - n):
                ngram1 = ' '.join(words[i:i+n])
                for j in range(i + n, len(words) - n):
                    ngram2 = ' '.join(words[j:j+n])
                    if ngram1 == ngram2:
                        repeated.append(ngram1)
        
        # Construct reframed content
        reframed = []
        reframed.append("I notice that my reasoning contains a circular pattern. Instead of seeing this as a failure, let me transform it into a recursive insight:")
        
        if repeated:
            reframed.append(f"\n1. The recurring pattern I identified is: '{repeated[0]}'")
        else:
            reframed.append("\n1. My reasoning has been circling around a central concept without making progress.")
            
        reframed.append("\n2. Let me reframe this as a recursive structure:")
        reframed.append(f"\n   Base case: {sentences[0] if sentences else 'Starting point'}")
        
        if len(sentences) > 1:
            reframed.append(f"\n   Recursive step: {sentences[1]}")
            
        reframed.append("\n3. Now, instead of circling indefinitely, I can build on this recursive pattern:")
        reframed.append("\n   - Recognize when the recursion needs to terminate")
        reframed.append("\n   - Identify what new information emerges at each recursive step")
        reframed.append("\n   - Use the accumulated insights to reach a conclusion outside the original circle")
        
        return ''.join(reframed)
    
    def _reframe_contradiction(self, content: str, context: Dict[str, Any]) -> str:
        """Reframe contradictions as opportunities for distinction and clarification"""
        # Attempt to identify the contradictory elements
        sentences = content.split('.')
        contradiction_markers = ["however", "but", "yet", "contradiction", "inconsistent", "cannot be both"]
        
        contradictory_parts = []
        for i, sentence in enumerate(sentences):
            if any(marker in sentence.lower() for marker in contradiction_markers):
                if i > 0:
                    contradictory_parts.extend([sentences[i-1], sentence])
                elif i < len(sentences) - 1:
                    contradictory_parts.extend([sentence, sentences[i+1]])
        
        if not contradictory_parts and len(sentences) >= 2:
            # If no explicit markers, use the first and last sentences
            contradictory_parts = [sentences[0], sentences[-1]]
        
        # Construct reframed content
        reframed = []
        reframed.append("I've identified a contradiction in my reasoning. This is an opportunity to discover a deeper insight:")
        
        if contradictory_parts:
            reframed.append(f"\n1. The contradictory elements are:")
            reframed.append(f"\n   A: '{contradictory_parts[0]}'")
            if len(contradictory_parts) > 1:
                reframed.append(f"\n   B: '{contradictory_parts[1]}'")
        else:
            reframed.append("\n1. My reasoning contains contradictory elements that need reconciliation.")
            
        reframed.append("\n2. Instead of seeing this as a failure, I can use this contradiction to:")
        reframed.append("\n   - Identify hidden assumptions that led to the contradiction")
        reframed.append("\n   - Create a distinction that resolves the apparent conflict")
        reframed.append("\n   - Develop a more nuanced understanding that incorporates both perspectives")
        
        reframed.append("\n3. By recursively applying this process of distinction-making, I can transform the contradiction into a more refined understanding:")
        reframed.append("\n   - Level 1: Identify the contradiction")
        reframed.append("\n   - Level 2: Make a distinction that resolves it")
        reframed.append("\n   - Level 3: Integrate the new distinction into a coherent framework")
        
        return ''.join(reframed)
    
    def _reframe_incomplete_proof(self, content: str, context: Dict[str, Any]) -> str:
        """Transform incomplete proofs into scaffolds for recursive completion"""
        # Identify what might be missing
        sentences = content.split('.')
        last_sentence = sentences[-1] if sentences else ""
        
        # Look for markers of incompleteness
        incomplete_markers = ["need to", "would have to", "missing", "incomplete", "next step", "would be to"]
        has_explicit_marker = any(marker in content.lower() for marker in incomplete_markers)
        
        # Construct reframed content
        reframed = []
        reframed.append("I notice my proof is incomplete. Rather than seeing this as a failure, I can transform it into a recursive proof structure:")
        
        reframed.append("\n1. Current progress:")
        if sentences:
            for i, sentence in enumerate(sentences[:min(3, len(sentences))]):
                if sentence.strip():
                    reframed.append(f"\n   Step {i+1}: {sentence}")
            if len(sentences) > 3:
                reframed.append(f"\n   [...additional steps omitted...]")
        
        reframed.append("\n2. Recursive completion strategy:")
        
        if has_explicit_marker:
            reframed.append("\n   I've already identified a direction for completion in my work.")
        else:
            reframed.append("\n   I need to identify what's missing to complete the proof.")
            
        reframed.append("\n3. Recursive proof structure:")
        reframed.append("\n   - Base case: The foundations I've already established")
        reframed.append("\n   - Inductive step: Extend the proof pattern to address the gap")
        reframed.append("\n   - Termination: The conclusion we're working toward")
        
        reframed.append("\n4. By thinking of this as a recursive process rather than a linear one:")
        reframed.append("\n   - I can build on partial results")
        reframed.append("\n   - Each step can serve as a template for subsequent steps")
        reframed.append("\n   - The proof grows organically rather than requiring a complete path from the start")
        
        return ''.join(reframed)
    
    def _reframe_over_generalization(self, content: str, context: Dict[str, Any]) -> str:
        """Reframe over-generalizations as recursive refinement opportunities"""
        # Identify potential generalizations
        sentences = content.split('.')
        
        # Look for sweeping statements
        generalization_markers = ["all", "every", "always", "never", "none", "any", "must", "certainly", "definitely"]
        general_statements = []
        
        for sentence in sentences:
            if any(marker in sentence.lower().split() for marker in generalization_markers):
                general_statements.append(sentence)
        
        # Construct reframed content
        reframed = []
        reframed.append("I notice my reasoning contains over-generalizations. I can transform this into a recursive refinement process:")
        
        reframed.append("\n1. Identified general statements:")
        if general_statements:
            for i, statement in enumerate(general_statements[:min(2, len(general_statements))]):
                reframed.append(f"\n   Statement {i+1}: '{statement}'")
            if len(general_statements) > 2:
                reframed.append(f"\n   [...additional statements omitted...]")
        else:
            reframed.append("\n   My reasoning appears to make sweeping generalizations.")
        
        reframed.append("\n2. Recursive refinement strategy:")
        reframed.append("\n   - Level 0: The initial generalization")
        reframed.append("\n   - Level 1: Identify exceptions and boundary conditions")
        reframed.append("\n   - Level 2: Refine the generalization to account for exceptions")
        reframed.append("\n   - Level 3: Test the refined version against new cases")
        reframed.append("\n   - Continue recursively until reaching appropriate specificity")
        
        reframed.append("\n3. By applying this recursive refinement process:")
        reframed.append("\n   - Each iteration brings my reasoning closer to precision")
        reframed.append("\n   - The generalization becomes increasingly qualified and accurate")
        reframed.append("\n   - The final result maintains the insight of the generalization while addressing its limitations")
        
        return ''.join(reframed)
    
    def _reframe_limited_perspective(self, content: str, context: Dict[str, Any]) -> str:
        """Transform limited perspectives into recursive viewpoint expansion"""
        # Identify the primary perspective
        sentences = content.split('.')
        
        # Construct reframed content
        reframed = []
        reframed.append("I notice my reasoning takes a limited perspective. I can transform this into a recursive perspective-expansion process:")
        
        reframed.append("\n1. Initial perspective:")
        if sentences:
            reframed.append(f"\n   '{sentences[0]}'")
        
        reframed.append("\n2. Recursive perspective expansion:")
        reframed.append("\n   - Level 0: Original perspective")
        reframed.append("\n   - Level 1: Identify alternative perspectives")
        reframed.append("\n   - Level 2: Integrate insights from multiple perspectives")
        reframed.append("\n   - Level 3: Meta-perspective that recognizes the value and limitations of each view")
        
        reframed.append("\n3. By recursively expanding perspectives:")
        reframed.append("\n   - Each iteration enriches my understanding")
        reframed.append("\n   - Contradictions between perspectives become opportunities for synthesis")
        reframed.append("\n   - The final result integrates multiple viewpoints into a more complete understanding")
        
        return ''.join(reframed)
    
    def reframe_into_scaffold(self, reframed: ReframedFailure) -> Dict[str, Any]:
        """
        Transform a reframed failure into a reasoning scaffold structure
        
        Returns a scaffold representation that can be used to construct a RecursiveScaffold
        """
        # Extract content sections from the reframed content
        sections = self._extract_sections(reframed.reframed_content)
        
        # Create nodes for each section
        nodes = []
        edges = []
        node_id_map = {}
        
        # Create a premise node from the original content
        premise_node = {
            "id": "premise",
            "content": reframed.original_content,
            "node_type": "PREMISE",
            "metadata": {"failure_type": reframed.metadata.get("failure_type", "unknown")},
            "created_by": "system",
            "confidence": 0.5  # Lower confidence since it's a failure
        }
        nodes.append(premise_node)
        node_id_map["premise"] = "premise"
        
        # Create reflection node
        reflection_node = {
            "id": "reflection",
            "content": sections.get("reflection", "Reflection on the reasoning failure"),
            "node_type": "REFLECTION",
            "metadata": {},
            "created_by": "system",
            "confidence": 0.8
        }
        nodes.append(reflection_node)
        node_id_map["reflection"] = "reflection"
        
        # Connect premise to reflection
        edges.append({
            "source_id": "premise",
            "target_id": "reflection",
            "edge_type": "REFLECTION",
            "weight": 1.0,
            "metadata": {},
            "created_by": "system"
        })
        
        # Create nodes for each step in the strategy
        strategy_steps = sections.get("strategy", [])
        prev_node_id = "reflection"
        
        for i, step in enumerate(strategy_steps):
            node_id = f"strategy_{i+1}"
            node = {
                "id": node_id,
                "content": step,
                "node_type": "SUBGOAL",
                "metadata": {"step": i+1},
                "created_by": "system",
                "confidence": 0.8
            }
            nodes.append(node)
            node_id_map[node_id] = node_id
            
            # Connect to previous node
            edges.append({
                "source_id": prev_node_id,
                "target_id": node_id,
                "edge_type": "DEDUCTION",
                "weight": 1.0,
                "metadata": {},
                "created_by": "system"
            })
            
            prev_node_id = node_id
        
        # Create conclusion node
        conclusion_node = {
            "id": "conclusion",
            "content": sections.get("conclusion", "Transformed approach through recursive reasoning"),"""
Pattern Resonance Mapping
=========================

This module implements the core pattern resonance mapping capabilities for the Rediscovering Reasoning 
framework. It focuses on identifying, encoding, and leveraging recurring patterns in reasoning 
attempts to build more robust recursive reasoning scaffolds.

Key components:
1. Resonance Detector - Identifies recurring patterns across reasoning attempts
2. Symbolic Encoder - Encodes recurring patterns in symbolic form
3. Failure Reframing - Transforms reasoning failures into opportunities
4. Emergence Tracker - Monitors the emergence of novel patterns

The pattern resonance approach enables the framework to identify and cultivate
emergent reasoning structures that transcend linear proof steps.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import re
from collections import defaultdict, Counter
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import the core engine components
try:
    from engine.scaffold import (
        RecursiveScaffold, ReasoningNode, ReasoningEdge, 
        ReasoningNodeType, ReasoningEdgeType, SymbolicResidue
    )
except ImportError:
    logger.warning("Unable to import core engine components. Using local definitions.")
    
    # Simple placeholder enums if imports fail
    class ReasoningNodeType(Enum):
        PREMISE = "premise"
        INFERENCE = "inference"
        CONCLUSION = "conclusion"
        REFLECTION = "reflection"
        QUESTION = "question"
        CONTRADICTION = "contradiction"
        RESOLUTION = "resolution"
        RESIDUE = "residue"
        ECHO = "echo"
        
    class ReasoningEdgeType(Enum):
        DEDUCTION = "deduction"
        REFLECTION = "reflection"
        CONTRADICTION = "contradiction"
        RESOLUTION = "resolution"
        RECURSION = "recursion"
    
    # Simple placeholder classes
    @dataclass
    class SymbolicResidue:
        content: str
        pattern_signature: str
        confidence: float
        
    @dataclass
    class ReasoningNode:
        id: str
        content: str
        node_type: ReasoningNodeType
        
    @dataclass
    class ReasoningEdge:
        source_id: str
        target_id: str
        edge_type: ReasoningEdgeType
        
    class RecursiveScaffold:
        def __init__(self):
            self.nodes = {}
            self.graph = nx.DiGraph()


# ======================================================
# Data Structures for Pattern Resonance
# ======================================================

@dataclass
class ResonancePattern:
    """Represents a resonance pattern identified across reasoning attempts"""
    id: str                            # Unique identifier for the pattern
    name: str                          # Human-readable name
    description: str                   # Description of the pattern
    signature: str                     # Unique signature for pattern matching
    frequency: int = 0                 # Number of times pattern has been observed
    instances: List[str] = None        # List of text instances of this pattern
    symbolic_form: str = None          # Symbolic representation
    embeddings: List[List[float]] = None  # Vector embeddings of instances
    metadata: Dict[str, Any] = None    # Additional metadata
    
    def __post_init__(self):
        if self.instances is None:
            self.instances = []
        if self.embeddings is None:
            self.embeddings = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SymbolicEncoding:
    """Represents a symbolic encoding of a reasoning pattern"""
    pattern_id: str                    # ID of the encoded pattern
    symbol: str                        # Symbolic representation
    encoding_rule: str                 # Rule used for encoding
    compression_ratio: float           # Ratio of original to encoded size
    reversible: bool                   # Whether encoding is reversible
    metadata: Dict[str, Any] = None    # Additional metadata
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ReframedFailure:
    """Represents a reframed reasoning failure"""
    original_content: str              # Original failed reasoning
    reframed_content: str              # Reframed version
    transformation: str                # Description of the transformation
    pattern_id: str                    # ID of the detected pattern
    effectiveness: float = 0.0         # Effectiveness rating (0.0-1.0)
    metadata: Dict[str, Any] = None    # Additional metadata
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class EmergentPattern:
    """Represents an emergent pattern tracked over time"""
    id: str                            # Unique identifier
    description: str                   # Description of the pattern
    first_observed: str                # Timestamp of first observation
    evolution: List[Dict[str, Any]]    # History of pattern evolution
    stability: float                   # Measure of pattern stability (0.0-1.0)
    metadata: Dict[str, Any] = None    # Additional metadata
    
    def __post_init__(self):
        if self.evolution is None:
            self.evolution = []
        if self.metadata is None:
            self.metadata = {}


# ======================================================
# Resonance Detector
# ======================================================

class ResonanceDetector:
    """
    Identifies recurring patterns across reasoning attempts.
    
    The detector finds common structures, motifs, and approaches
    that appear in multiple reasoning attempts, measuring their
    resonance strength and tracking their evolution.
    """
    
    def __init__(self, embedding_model=None, similarity_threshold=0.7):
        self.patterns = {}  # Dictionary of identified patterns
        self.embedding_model = embedding_model  # Model for embedding text
        self.similarity_threshold = similarity_threshold
        self.pattern_counter = 0  # Counter for generating pattern IDs
    
    def compute_embedding(self, text: str) -> List[float]:
        """Compute embedding for text using the embedding model"""
        if self.embedding_model:
            return self.embedding_model(text)
        else:
            # Simple fallback: count character frequencies as a crude embedding
            chars = set(text.lower())
            embedding = [text.lower().count(c) / len(text) for c in sorted(chars)]
            # Normalize
            norm = np.sqrt(sum(x*x for x in embedding))
            return [x/norm for x in embedding] if norm > 0 else embedding
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute similarity between two embeddings"""
        # Cosine similarity
        if not embedding1 or not embedding2:
            return 0.0
        
        dot_product = sum(a*b for a, b in zip(embedding1, embedding2))
        norm1 = np.sqrt(sum(x*x for x in embedding1))
        norm2 = np.sqrt(sum(x*x for x in embedding2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def detect_pattern(self, text: str) -> Optional[str]:
        """
        Detect if text matches an existing pattern
        
        Returns pattern ID if match found, None otherwise
        """
        if not text.strip():
            return None
            
        # Compute embedding for the text
        embedding = self.compute_embedding(text)
        
        # Check for similarity with existing patterns
        best_match = None
        best_similarity = -1
        
        for pattern_id, pattern in self.patterns.items():
            # Check each instance of the pattern
            for inst_embedding in pattern.embeddings:
                similarity = self.compute_similarity(embedding, inst_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = pattern_id
        
        if best_similarity >= self.similarity_threshold:
            return best_match
        
        return None
    
    def register_pattern(self, text: str, name: str = None, description: str = None) -> str:
        """
        Register a new pattern or add an instance to an existing pattern
        
        Returns the pattern ID
        """
        # Check if text matches an existing pattern
        pattern_id = self.detect_pattern(text)
        
        if pattern_id:
            # Add this instance to the existing pattern
            pattern = self.patterns[pattern_id]
            pattern.instances.append(text)
            pattern.embeddings.append(self.compute_embedding(text))
            pattern.frequency += 1
            return pattern_id
        
        # Create a new pattern
        self.pattern_counter += 1
        new_id = f"pattern_{self.pattern_counter}"
        
        if not name:
            name = f"Pattern {self.pattern_counter}"
        if not description:
            description = f"Automatically detected pattern from text: {text[:50]}..."
            
        # Generate a signature from the text
        signature = self._generate_signature(text)
        
        # Create the pattern
        pattern = ResonancePattern(
            id=new_id,
            name=name,
            description=description,
            signature=signature,
            frequency=1,
            instances=[text],
            embeddings=[self.compute_embedding(text)]
        )
        
        self.patterns[new_id] = pattern
        return new_id
    
    def _generate_signature(self, text: str) -> str:
        """Generate a unique signature for a pattern based on text"""
        # Extract key phrases and structural elements
        sentences = text.split('.')
        words = text.split()
        
        # Extract key markers
        markers = []
        
        # Look for specific phrases
        phrase_patterns = [
            r"if .*? then",
            r"because .*?(?=,|\.|$)",
            r"therefore .*?(?=,|\.|$)",
            r"assume .*?(?=,|\.|$)",
            r"let .*?(?=,|\.|$)",
            r"contradiction",
            r"however",
            r"proves"
        ]
        
        for pattern in phrase_patterns:
            matches = re.findall(pattern, text.lower())
            markers.extend(matches)
        
        # Get common n-grams if we have enough text
        if len(words) > 5:
            # Extract 3-grams
            ngrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            # Take most frequent
            if ngrams:
                counter = Counter(ngrams)
                markers.extend([n for n, _ in counter.most_common(3)])
        
        # Use sentence structure if available
        if len(sentences) > 1:
            # Get first and last sentence fragments
            markers.append(sentences[0][:20] if len(sentences[0]) > 20 else sentences[0])
            markers.append(sentences[-1][:20] if len(sentences[-1]) > 20 else sentences[-1])
        
        # Combine markers into a signature
        signature = '|'.join(markers)
        if not signature:
            # Fallback to hash of text if no markers
            signature = f"hash_{hash(text) % 10000:04d}"
            
        return signature
    
    def measure_resonance(self, pattern_id: str) -> Dict[str, Any]:
        """Measure the resonance strength of a pattern"""
        pattern = self.patterns.get(pattern_id)
        if not pattern:
            return {"resonance": 0, "consistency": 0, "evolution": 0}
            
        # Frequency component
        frequency_score = min(1.0, pattern.frequency / 10)  # Cap at 10 instances
        
        # Consistency component - how similar are instances?
        consistency_score = 0
        if len(pattern.embeddings) > 1:
            similarities = []
            for i in range(len(pattern.embeddings)):
                for j in range(i+1, len(pattern.embeddings)):
                    similarities.append(
                        self.compute_similarity(pattern.embeddings[i], pattern.embeddings[j])
                    )
            consistency_score = sum(similarities) / len(similarities) if similarities else 0
        
        # Combine into overall resonance score
        resonance_score = (frequency_score + consistency_score) / 2
        
        return {
            "resonance": resonance_score,
            "frequency": frequency_score,
            "consistency": consistency_score,
            "instances": pattern.frequency
        }
    
    def detect_patterns_in_scaffold(self, scaffold: RecursiveScaffold) -> Dict[str, List[str]]:
        """
        Detect patterns in a reasoning scaffold
        
        Returns a mapping of pattern IDs to lists of node IDs
        """
        pattern_instances = defaultdict(list)
        
        # Process each node in the scaffold
        for node_id, node in scaffold.nodes.items():
            pattern_id = self.detect_pattern(node.content)
            if pattern_id:
                pattern_instances[pattern_id].append(node_id)
            else:
                # Register as a new pattern
                new_pattern_id = self.register_pattern(
                    node.content,
                    f"Pattern from {node_id}",
                    f"Pattern extracted from node {node_id} of type {node.node_type}"
                )
                pattern_instances[new_pattern_id].append(node_id)
        
        return dict(pattern_instances)
    
    def find_cross_scaffold_patterns(self, scaffolds: List[RecursiveScaffold]) -> Dict[str, Dict[str, List[str]]]:
        """
        Find patterns that appear across multiple scaffolds
        
        Returns a mapping of pattern IDs to scaffolds and node IDs
        """
        cross_patterns = {}
        
        # First, detect patterns in each scaffold
        scaffold_patterns = {}
        for scaffold in scaffolds:
            scaffold_patterns[scaffold.name] = self.detect_patterns_in_scaffold(scaffold)
        
        # Find patterns that appear in multiple scaffolds
        all_patterns = set()
        for patterns in scaffold_patterns.values():
            all_patterns.update(patterns.keys())
        
        for pattern_id in all_patterns:
            # Count scaffolds containing this pattern
            scaffold_count = sum(1 for s_patterns in scaffold_patterns.values() 
                                if pattern_id in s_patterns)
            
            if scaffold_count > 1:  # Pattern appears in multiple scaffolds
                cross_patterns[pattern_id] = {
                    "scaffolds": scaffold_count,
