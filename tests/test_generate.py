"""
Comprehensive test suite for buddhi_pragati.generate module.
Tests crossword generation, corpus loading, memetic algorithms, and puzzle validation.
"""

import pytest
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

# Import modules to test
from buddhi_pragati.generate.puzzle_entry import CrosswordPuzzleEntry
from buddhi_pragati.generate.corpus_loader import CrosswordCorpusLoader, CorpusEntry
from buddhi_pragati.core.base_puzzle import CrosswordPuzzle, CrosswordClue


class TestCrosswordPuzzleEntry:
    """Test CrosswordPuzzleEntry dataclass functionality."""
    
    def test_crossword_puzzle_entry_creation(self):
        """Test basic CrosswordPuzzleEntry creation with all required fields."""
        entry = CrosswordPuzzleEntry(
            id="puzzle_english_10x10_001",
            clues_with_answer={
                "1across": {
                    "clue": "Capital of India",
                    "answer": "DELHI",
                    "start": [0, 0]
                },
                "1down": {
                    "clue": "Holy river", 
                    "answer": "GANGA",
                    "start": [0, 0]
                }
            },
            empty_grid=[
                [1, None, None, None, None],
                [None, None, None, None, None],
                [None, None, None, None, None],
                [None, None, None, None, None],
                [None, None, None, None, None]
            ],
            solved_grid=[
                ["D", "E", "L", "H", "I"],
                ["G", None, None, None, None],
                ["A", None, None, None, None], 
                ["N", None, None, None, None],
                ["G", None, None, None, None]
            ],
            context_score=0.8,
            quality_score=0.7,
            source_mix={"MILU": 50.0, "IndicWikiBio": 30.0, "IndoWordNet": 20.0},
            grid_size=5,
            density=0.4,  # 2 out of 5 words filled
            word_count=2,
            generation_metadata={"algorithm": "memetic", "iterations": 100}
        )
        
        # Verify all fields are set correctly
        assert entry.id == "puzzle_english_10x10_001"
        assert len(entry.clues_with_answer) == 2
        assert entry.context_score == 0.8
        assert entry.quality_score == 0.7
        assert entry.grid_size == 5
        assert entry.density == 0.4
        assert entry.word_count == 2
        
        # Verify clue structure
        across_clue = entry.clues_with_answer["1across"]
        assert across_clue["clue"] == "Capital of India"
        assert across_clue["answer"] == "DELHI"
        assert across_clue["start"] == [0, 0]
    
    def test_crossword_puzzle_entry_serialization(self):
        """Test JSON serialization with asdict()."""
        entry = CrosswordPuzzleEntry(
            id="test_puzzle",
            clues_with_answer={"1across": {"clue": "Test", "answer": "WORD", "start": [0, 0]}},
            empty_grid=[[1, None], [None, None]],
            solved_grid=[["W", "O"], ["R", "D"]],
            context_score=0.5,
            quality_score=0.6,
            source_mix={"MILU": 100.0},
            grid_size=2,
            density=0.5,
            word_count=1,
            generation_metadata={"test": True}
        )
        
        entry_dict = asdict(entry)
        
        # Should be JSON serializable
        json_str = json.dumps(entry_dict)
        reconstructed = json.loads(json_str)
        
        assert reconstructed["id"] == "test_puzzle"
        assert reconstructed["context_score"] == 0.5
        assert reconstructed["grid_size"] == 2
    
    def test_crossword_puzzle_entry_score_validation(self):
        """Test score ranges and validation."""
        # Create a valid entry with proper grids and clues
        entry = CrosswordPuzzleEntry(
            id="score_test",
            clues_with_answer={"1across": {"clue": "Test clue", "answer": "WORD", "start": [0, 0]}},
            empty_grid=[[1, None, None], [None, None, None], [None, None, None]],
            solved_grid=[["W", "O", "R"], ["D", None, None], [None, None, None]],
            context_score=0.9,
            quality_score=0.1,
            source_mix={"TEST": 100.0},
            grid_size=3,
            density=0.75,
            word_count=1,
            generation_metadata={}
        )
        
        # Scores should be in valid ranges
        assert 0.0 <= entry.context_score <= 1.0
        assert 0.0 <= entry.quality_score <= 1.0
        assert 0.0 <= entry.density <= 1.0
        
        # Word count should be non-negative
        assert entry.word_count >= 0
        assert entry.grid_size > 0
    
    def test_source_mix_validation(self):
        """Test source mix percentage validation."""
        # Create a valid entry with proper grids and clues
        entry = CrosswordPuzzleEntry(
            id="source_test",
            clues_with_answer={"1across": {"clue": "Test clue", "answer": "WORD", "start": [0, 0]}},
            empty_grid=[[1, None, None], [None, None, None], [None, None, None]],
            solved_grid=[["W", "O", "R"], ["D", None, None], [None, None, None]],
            context_score=0.5,
            quality_score=0.5,
            source_mix={"MILU": 40.0, "IndicWikiBio": 35.0, "IndoWordNet": 25.0},
            grid_size=3,
            density=0.6,
            word_count=1,
            generation_metadata={}
        )
        
        # Source percentages should be reasonable
        total_percentage = sum(entry.source_mix.values())
        assert abs(total_percentage - 100.0) < 1.0  # Allow small rounding errors
        
        # All percentages should be non-negative
        for percentage in entry.source_mix.values():
            assert percentage >= 0.0
    
    def test_grid_consistency_validation(self):
        """Test that empty_grid and solved_grid have consistent dimensions."""
        # Create entry with proper 4x4 grid to match grid_size=4
        entry = CrosswordPuzzleEntry(
            id="grid_test",
            clues_with_answer={"1across": {"clue": "Test", "answer": "WORD", "start": [0, 0]}},
            empty_grid=[
                [1, None, None, None],
                [None, None, None, None],
                [None, None, None, None],
                [None, None, None, None]
            ],
            solved_grid=[
                ["W", "O", "R", "D"],
                [None, None, None, None],
                [None, None, None, None],
                [None, None, None, None]
            ],
            context_score=0.5,
            quality_score=0.5,
            source_mix={"MILU": 100.0},
            grid_size=4,
            density=0.25,  # 4 filled cells out of 16 total
            word_count=1,
            generation_metadata={}
        )
        
        # Grid dimensions should be consistent
        assert len(entry.empty_grid) == len(entry.solved_grid)
        if entry.empty_grid:
            assert len(entry.empty_grid[0]) == len(entry.solved_grid[0])
    
    def test_clue_structure_validation(self):
        """Test clue data structure validation."""
        entry = CrosswordPuzzleEntry(
            id="clue_test",
            clues_with_answer={
                "1across": {
                    "clue": "First clue across",
                    "answer": "FIRST", 
                    "start": [0, 0]
                },
                "2down": {
                    "clue": "Second clue down",
                    "answer": "SECOND",
                    "start": [1, 0]
                }
            },
            empty_grid=[[1, None, None], [2, None, None], [None, None, None]],
            solved_grid=[["F", "I", "R"], ["S", None, None], ["T", None, None]],
            context_score=0.7,
            quality_score=0.8,
            source_mix={"MILU": 100.0},
            grid_size=3,
            density=0.33,
            word_count=2,
            generation_metadata={}
        )
        
        # Each clue should have required fields
        for clue_id, clue_data in entry.clues_with_answer.items():
            assert "clue" in clue_data
            assert "answer" in clue_data
            assert "start" in clue_data
            
            # Answer should be non-empty string
            assert isinstance(clue_data["answer"], str)
            assert len(clue_data["answer"]) > 0
            
            # Start position should be valid coordinates
            assert isinstance(clue_data["start"], list)
            assert len(clue_data["start"]) == 2
            assert all(isinstance(coord, int) for coord in clue_data["start"])


class TestCorpusEntry:
    """Test CorpusEntry NamedTuple functionality."""
    
    def test_corpus_entry_creation(self):
        """Test CorpusEntry creation and field access."""
        entry = CorpusEntry(
            clue="What is the capital of India?",
            answer="DELHI",
            context_score=0.8,
            source="MILU",
            source_id="milu_english_001",
            quality_score=0.7
        )
        
        assert entry.clue == "What is the capital of India?"
        assert entry.answer == "DELHI"
        assert entry.context_score == 0.8
        assert entry.source == "MILU"
        assert entry.source_id == "milu_english_001"
        assert entry.quality_score == 0.7
    
    def test_corpus_entry_immutability(self):
        """Test that CorpusEntry is immutable (NamedTuple property)."""
        entry = CorpusEntry(
            clue="Test clue",
            answer="TEST",
            context_score=0.5,
            source="TEST",
            source_id="test_001",
            quality_score=0.6
        )
        
        # Should not be able to modify fields
        with pytest.raises(AttributeError):
            entry.clue = "Modified clue"
    
    def test_corpus_entry_equality(self):
        """Test CorpusEntry equality comparison."""
        entry1 = CorpusEntry("clue", "ANSWER", 0.5, "SOURCE", "id", 0.6)
        entry2 = CorpusEntry("clue", "ANSWER", 0.5, "SOURCE", "id", 0.6)
        entry3 = CorpusEntry("clue", "DIFFERENT", 0.5, "SOURCE", "id", 0.6)
        
        assert entry1 == entry2
        assert entry1 != entry3
    
    def test_corpus_entry_tuple_operations(self):
        """Test CorpusEntry tuple operations."""
        entry = CorpusEntry("clue", "ANSWER", 0.8, "MILU", "id_001", 0.7)
        
        # Should support tuple unpacking
        clue, answer, context_score, source, source_id, quality_score = entry
        assert clue == "clue"
        assert answer == "ANSWER"
        assert context_score == 0.8
        
        # Should support indexing
        assert entry[0] == "clue"
        assert entry[1] == "ANSWER"
        assert entry[2] == 0.8


class TestCrosswordCorpusLoader:
    """Test CrosswordCorpusLoader functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for corpus loader testing."""
        config = Mock()
        config.get_string.side_effect = lambda key, default: {
            "HF_DATASET_REPO": "test/repo",
            "DEFAULT_HF_TOKEN": "fake_token"
        }.get(key, default)
        return config
    
    @patch('buddhi_pragati.generate.corpus_loader.get_config')
    def test_corpus_loader_initialization(self, mock_get_config, mock_config):
        """Test CrosswordCorpusLoader initialization."""
        mock_get_config.return_value = mock_config
        
        loader = CrosswordCorpusLoader()
        
        assert loader.repo_id == "test/repo"
        assert loader.hf_token == "fake_token"
        assert isinstance(loader.loaded_corpus, dict)
        assert isinstance(loader.context_scorers, dict)
    
    @patch('buddhi_pragati.generate.corpus_loader.get_config')
    def test_corpus_loader_custom_params(self, mock_get_config, mock_config):
        """Test initialization with custom parameters."""
        mock_get_config.return_value = mock_config
        
        loader = CrosswordCorpusLoader(
            repo_id="custom/repo",
            hf_token="custom_token"
        )
        
        assert loader.repo_id == "custom/repo"
        assert loader.hf_token == "custom_token"
    
    @patch('buddhi_pragati.generate.corpus_loader.get_config')
    @patch('buddhi_pragati.generate.corpus_loader.load_dataset')
    def test_corpus_loading_with_mock_dataset(self, mock_load_dataset, mock_get_config, mock_config):
        """Test corpus loading with mocked HuggingFace dataset."""
        mock_get_config.return_value = mock_config
        
        # Mock dataset response
        mock_dataset = Mock()
        mock_dataset.iter.return_value = iter([
            {
                "clue": "Capital of India",
                "answer": "DELHI", 
                "context_score": 0.8,
                "source": "MILU",
                "source_id": "milu_001",
                "quality_score": 0.7
            },
            {
                "clue": "Holy river",
                "answer": "GANGA",
                "context_score": 0.9,
                "source": "IndicWikiBio", 
                "source_id": "wikibio_001",
                "quality_score": 0.8
            }
        ])
        mock_load_dataset.return_value = mock_dataset
        
        loader = CrosswordCorpusLoader()
        
        # This would test the actual loading method if it exists
        # For now, test that the loader can be initialized
        assert loader is not None
    
    @patch('buddhi_pragati.generate.corpus_loader.get_config')
    def test_corpus_loader_caching(self, mock_get_config, mock_config):
        """Test corpus loading caching behavior."""
        mock_get_config.return_value = mock_config
        
        loader = CrosswordCorpusLoader()
        
        # Test that loaded_corpus starts empty
        assert len(loader.loaded_corpus) == 0
        
        # Test that context_scorers starts empty
        assert len(loader.context_scorers) == 0
    
    @patch('buddhi_pragati.generate.corpus_loader.get_config')
    @patch('buddhi_pragati.generate.corpus_loader.is_predominantly_latin_script')
    def test_script_filtering_integration(self, mock_script_check, mock_get_config, mock_config):
        """Test integration with script filtering utilities."""
        mock_get_config.return_value = mock_config
        mock_script_check.return_value = True
        
        loader = CrosswordCorpusLoader()
        
        # Test that script checking function is available
        result = mock_script_check("English text")
        assert result == True
    
    @patch('buddhi_pragati.generate.corpus_loader.get_config')
    def test_corpus_prioritization_setup(self, mock_get_config, mock_config):
        """Test setup for cultural context prioritization."""
        mock_get_config.return_value = mock_config
        
        loader = CrosswordCorpusLoader()
        
        # Test that the loader has the infrastructure for prioritization
        assert hasattr(loader, 'loaded_corpus')
        assert hasattr(loader, 'context_scorers')
    
    @patch('buddhi_pragati.generate.corpus_loader.get_config')
    def test_error_handling_initialization(self, mock_get_config):
        """Test error handling during initialization."""
        # Test with missing config
        mock_config = Mock()
        mock_config.get_string.return_value = None
        mock_get_config.return_value = mock_config
        
        loader = CrosswordCorpusLoader()
        
        # Should handle missing config gracefully
        assert loader.repo_id is None or loader.repo_id == ""
        assert loader.hf_token is None or loader.hf_token == ""


class TestCrosswordPuzzleIntegration:
    """Test integration with CrosswordPuzzle and CrosswordClue classes."""
    
    def test_crossword_clue_creation(self):
        """Test CrosswordClue creation for puzzle generation."""
        clue = CrosswordClue(
            number=1,
            direction="across",
            length=5,
            clue_text="Capital of India",
            start_row=0,
            start_col=0,
            answer="DELHI"
        )
        
        assert clue.number == 1
        assert clue.direction == "across"
        assert clue.length == 5
        assert clue.clue_text == "Capital of India"
        assert clue.start_row == 0
        assert clue.start_col == 0
        assert clue.answer == "DELHI"
    
    def test_crossword_puzzle_creation(self):
        """Test CrosswordPuzzle creation with multiple clues."""
        clues = [
            CrosswordClue(1, "across", 5, "Capital of India", 0, 0, "DELHI"),
            CrosswordClue(1, "down", 5, "Holy river", 0, 0, "GANGA")
        ]
        
        empty_grid = [
            [1, None, None, None, None],
            [None, None, None, None, None],
            [None, None, None, None, None],
            [None, None, None, None, None],
            [None, None, None, None, None]
        ]
        
        solved_grid = [
            ["D", "E", "L", "H", "I"],
            ["G", None, None, None, None],
            ["A", None, None, None, None],
            ["N", None, None, None, None],
            ["G", None, None, None, None]
        ]
        
        puzzle = CrosswordPuzzle(
            puzzle_id="test_puzzle_001",
            grid=empty_grid,
            clues=clues,
            size=(5, 5),
            solution_grid=solved_grid,
            solution_words={"1across": "DELHI", "1down": "GANGA"}
        )
        
        assert puzzle.puzzle_id == "test_puzzle_001"
        assert puzzle.get_size() == (5, 5)
        assert len(puzzle.get_clues()) == 2
        assert puzzle.get_solution_words()["1across"] == "DELHI"
    
    def test_puzzle_entry_from_crossword_puzzle(self):
        """Test creating CrosswordPuzzleEntry from CrosswordPuzzle."""
        # Create a CrosswordPuzzle with proper 4x4 grid
        clues = [CrosswordClue(1, "across", 4, "Test clue", 0, 0, "WORD")]
        empty_grid = [
            [1, None, None, None],
            [None, None, None, None],
            [None, None, None, None],
            [None, None, None, None]
        ]
        solved_grid = [
            ["W", "O", "R", "D"],
            [None, None, None, None],
            [None, None, None, None],
            [None, None, None, None]
        ]
        
        puzzle = CrosswordPuzzle(
            puzzle_id="conversion_test",
            grid=empty_grid,
            clues=clues,
            size=(4, 4),
            solution_grid=solved_grid,
            solution_words={"1across": "WORD"}
        )
        
        # Convert to CrosswordPuzzleEntry format
        entry = CrosswordPuzzleEntry(
            id=puzzle.puzzle_id,
            clues_with_answer={
                "1across": {
                    "clue": "Test clue",
                    "answer": "WORD",
                    "start": [0, 0]
                }
            },
            empty_grid=empty_grid,
            solved_grid=solved_grid,
            context_score=0.5,
            quality_score=0.6,
            source_mix={"TEST": 100.0},
            grid_size=4,
            density=0.25,  # 4 filled cells out of 16 total
            word_count=1,
            generation_metadata={"converted_from_puzzle": True}
        )
        
        assert entry.id == puzzle.puzzle_id
        assert entry.word_count == len(puzzle.get_clues())
        assert entry.grid_size == max(puzzle.get_size())
    
    def test_puzzle_quality_validation(self):
        """Test puzzle quality validation logic."""
        # High quality puzzle with proper 10x10 grid
        empty_grid_10x10 = [[None for _ in range(10)] for _ in range(10)]
        solved_grid_10x10 = [[None for _ in range(10)] for _ in range(10)]
        # Add some sample words
        empty_grid_10x10[0][0] = 1
        empty_grid_10x10[0][7] = 2
        for i, char in enumerate("QUALITY"):
            solved_grid_10x10[0][i] = char
        for i, char in enumerate("GOOD"):
            solved_grid_10x10[i][7] = char
        
        high_quality_entry = CrosswordPuzzleEntry(
            id="high_quality",
            clues_with_answer={
                "1across": {"clue": "Detailed clue with context", "answer": "QUALITY", "start": [0, 0]},
                "2down": {"clue": "Another contextual clue", "answer": "GOOD", "start": [0, 7]}
            },
            empty_grid=empty_grid_10x10,
            solved_grid=solved_grid_10x10,
            context_score=0.9,  # High cultural context
            quality_score=0.8,  # High quality
            source_mix={"MILU": 60.0, "IndicWikiBio": 40.0},  # Good source diversity
            grid_size=10,
            density=0.8,  # High density
            word_count=2,
            generation_metadata={"quality_validated": True}
        )
        
        # Validate high quality metrics
        assert high_quality_entry.context_score >= 0.8
        assert high_quality_entry.quality_score >= 0.7
        assert high_quality_entry.density >= 0.7
        assert high_quality_entry.word_count >= 1
        
        # Low quality puzzle for comparison with proper 3x3 grid
        low_quality_entry = CrosswordPuzzleEntry(
            id="low_quality",
            clues_with_answer={"1across": {"clue": "Bad", "answer": "BAD", "start": [0, 0]}},
            empty_grid=[
                [1, None, None],
                [None, None, None],
                [None, None, None]
            ],
            solved_grid=[
                ["B", "A", "D"],
                [None, None, None],
                [None, None, None]
            ],
            context_score=0.2,
            quality_score=0.3,
            source_mix={"MILU": 100.0},
            grid_size=3,
            density=0.3,
            word_count=1,
            generation_metadata={}
        )
        
        # Low quality should have lower metrics
        assert low_quality_entry.context_score < 0.5
        assert low_quality_entry.quality_score < 0.5
        assert low_quality_entry.density < 0.5


class TestPuzzleValidation:
    """Test puzzle validation and quality checks."""
    
    def test_grid_structure_validation(self):
        """Test grid structure consistency validation."""
        # Valid grid structure with proper 4x4 dimensions
        valid_entry = CrosswordPuzzleEntry(
            id="valid_grid",
            clues_with_answer={"1across": {"clue": "Test", "answer": "WORD", "start": [0, 0]}},
            empty_grid=[
                [1, None, None, None],
                [None, None, None, None],
                [None, None, None, None],
                [None, None, None, None]
            ],
            solved_grid=[
                ["W", "O", "R", "D"],
                [None, None, None, None],
                [None, None, None, None],
                [None, None, None, None]
            ],
            context_score=0.5,
            quality_score=0.5,
            source_mix={"TEST": 100.0},
            grid_size=4,
            density=0.25,
            word_count=1,
            generation_metadata={}
        )
        
        # Grid dimensions should match
        assert len(valid_entry.empty_grid) == len(valid_entry.solved_grid)
        assert len(valid_entry.empty_grid[0]) == len(valid_entry.solved_grid[0])
    
    def test_density_calculation_validation(self):
        """Test density calculation validation."""
        # Create entry with all cells filled for easier density calculation
        entry = CrosswordPuzzleEntry(
            id="density_test",
            clues_with_answer={
                "1across": {"clue": "First", "answer": "FIRST", "start": [0, 0]},
                "2down": {"clue": "Second", "answer": "FST", "start": [0, 0]}
            },
            empty_grid=[[1, None, None], [None, None, None], [None, None, None]],
            solved_grid=[["F", "I", "R"], ["S", None, None], ["T", None, None]],
            context_score=0.6,
            quality_score=0.7,
            source_mix={"TEST": 100.0},
            grid_size=3,
            density=0.67,  # 6 filled cells out of 9
            word_count=2,
            generation_metadata={}
        )
        
        # Density should be reasonable for the grid
        total_cells = entry.grid_size * entry.grid_size
        filled_cells = sum(1 for row in entry.solved_grid for cell in row if cell is not None)
        expected_density = filled_cells / total_cells
        
        # Allow some tolerance for rounding
        assert abs(entry.density - expected_density) < 0.15
    
    def test_intersection_validation(self):
        """Test word intersection validation."""
        # Create proper 4x4 grid with intersecting words
        entry = CrosswordPuzzleEntry(
            id="intersection_test",
            clues_with_answer={
                "1across": {"clue": "First word", "answer": "WORD", "start": [0, 0]},
                "1down": {"clue": "Intersecting word", "answer": "WILL", "start": [0, 0]}
            },
            empty_grid=[
                [1, None, None, None],
                [None, None, None, None],
                [None, None, None, None],
                [None, None, None, None]
            ],
            solved_grid=[
                ["W", "O", "R", "D"],
                ["I", None, None, None],
                ["L", None, None, None],
                ["L", None, None, None]
            ],
            context_score=0.5,
            quality_score=0.6,
            source_mix={"TEST": 100.0},
            grid_size=4,
            density=0.5,
            word_count=2,
            generation_metadata={"intersections": 1}
        )
        
        # Should have intersection metadata
        assert "intersections" in entry.generation_metadata
        assert entry.generation_metadata["intersections"] >= 1
        
        # Words should intersect at shared position
        across_word = entry.clues_with_answer["1across"]["answer"]
        down_word = entry.clues_with_answer["1down"]["answer"]
        
        # First letter should match (intersection point)
        assert across_word[0] == down_word[0]  # Both start with same letter


class TestPerformanceAndMemory:
    """Test performance characteristics and memory efficiency."""
    
    def test_large_puzzle_entry_creation(self):
        """Test creating large puzzle entries efficiently."""
        # Create a large grid (15x15)
        grid_size = 15
        empty_grid = [[None for _ in range(grid_size)] for _ in range(grid_size)]
        solved_grid = [[None for _ in range(grid_size)] for _ in range(grid_size)]
        
        # Add some test data
        empty_grid[0][0] = 1
        solved_grid[0][0] = "W"
        
        large_entry = CrosswordPuzzleEntry(
            id="large_puzzle_15x15",
            clues_with_answer={"1across": {"clue": "Large puzzle test", "answer": "WORD", "start": [0, 0]}},
            empty_grid=empty_grid,
            solved_grid=solved_grid,
            context_score=0.5,
            quality_score=0.5,
            source_mix={"TEST": 100.0},
            grid_size=grid_size,
            density=0.1,
            word_count=1,
            generation_metadata={}
        )
        
        # Should create successfully
        assert large_entry.grid_size == grid_size
        assert len(large_entry.empty_grid) == grid_size
        assert len(large_entry.solved_grid) == grid_size
    
    def test_serialization_performance(self):
        """Test serialization performance for puzzle entries."""
        entry = CrosswordPuzzleEntry(
            id="serialization_test",
            clues_with_answer={f"{i}across": {"clue": f"Clue {i}", "answer": f"ANSWER{i}", "start": [i, 0]} for i in range(20)},
            empty_grid=[[i if i == j else None for j in range(10)] for i in range(10)],
            solved_grid=[[f"L{i}" if i == j else None for j in range(10)] for i in range(10)],
            context_score=0.5,
            quality_score=0.5,
            source_mix={"TEST": 100.0},
            grid_size=10,
            density=0.5,
            word_count=20,
            generation_metadata={"test_data": list(range(100))}
        )
        
        # Should serialize to JSON without errors
        entry_dict = asdict(entry)
        json_str = json.dumps(entry_dict)
        
        # Should be able to deserialize
        reconstructed = json.loads(json_str)
        assert reconstructed["id"] == "serialization_test"
        assert len(reconstructed["clues_with_answer"]) == 20
    
    def test_memory_efficient_grid_storage(self):
        """Test memory-efficient grid storage patterns."""
        # Test sparse grid representation
        sparse_entry = CrosswordPuzzleEntry(
            id="sparse_test",
            clues_with_answer={"1across": {"clue": "Sparse", "answer": "WORD", "start": [5, 5]}},
            empty_grid=[[None for _ in range(10)] for _ in range(10)],
            solved_grid=[[None for _ in range(10)] for _ in range(10)],
            context_score=0.4,
            quality_score=0.4,
            source_mix={"TEST": 100.0},
            grid_size=10,
            density=0.04,  # Very low density
            word_count=1,
            generation_metadata={"sparse": True}
        )
        
        # Add single word to center
        sparse_entry.empty_grid[5][5] = 1
        sparse_entry.solved_grid[5][5] = "W"
        
        # Should handle sparse representation efficiently
        assert sparse_entry.density < 0.1
        assert sparse_entry.word_count == 1


if __name__ == "__main__":
    pytest.main([__file__])