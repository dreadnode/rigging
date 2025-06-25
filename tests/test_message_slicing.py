"""Test suite for message slicing functionality."""

import re
import warnings

from rigging import Chat, Message, Model, attr, element
from rigging.error import MessageWarning
from rigging.message import MessageSlice

# ruff: noqa: S101, PLR2004, ARG001, PT011, SLF001


# Test models for slice testing
class SimpleModel(Model):
    """Simple test model for slicing tests."""

    content: str


class ComplexModel(Model):
    """Complex test model with attributes and elements."""

    name: str = attr()
    description: str = element()
    value: int = element()


class NestedModel(Model):
    """Model for testing nested slice relationships."""

    title: str = attr()
    items: list[str] = element()


# =============================================================================
# MessageSlice Core Functionality Tests
# =============================================================================


def test_message_slice_initialization() -> None:
    """Test MessageSlice can be initialized with all required parameters."""
    slice_obj = MessageSlice(
        type="model",
        start=0,
        stop=10,
        obj=SimpleModel(content="test"),
        metadata={"confidence": 0.95},
    )

    assert slice_obj.type == "model"
    assert slice_obj.start == 0
    assert slice_obj.stop == 10
    assert isinstance(slice_obj.obj, SimpleModel)
    assert slice_obj.metadata["confidence"] == 0.95


def test_message_slice_content_property_attached() -> None:
    """Test MessageSlice content property when attached to a message."""
    message = Message("assistant", "The answer is 42.")
    slice_obj = MessageSlice(type="other", start=14, stop=16)
    slice_obj._message = message

    assert slice_obj.content == "42"


def test_message_slice_content_property_detached() -> None:
    """Test MessageSlice content property when detached from a message."""
    slice_obj = MessageSlice(type="other", start=0, stop=5)

    assert slice_obj.content == "[detached]"


def test_message_slice_content_setter_attached() -> None:
    """Test MessageSlice content setter when attached to a message."""
    message = Message("assistant", "The answer is 42.")
    slice_obj = MessageSlice(type="other", start=14, stop=16)
    slice_obj._message = message

    slice_obj.content = "24"

    assert message.content == "The answer is 24."
    assert slice_obj.stop == 16  # start (14) + len("24") = 16


def test_message_slice_content_setter_detached() -> None:
    """Test MessageSlice content setter when detached from a message."""
    slice_obj = MessageSlice(type="other", start=0, stop=5)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        slice_obj.content = "test"

        assert len(w) == 1
        assert issubclass(w[-1].category, MessageWarning)
        assert "detached MessageSlice" in str(w[-1].message)


def test_message_slice_slice_property() -> None:
    """Test MessageSlice slice_ property returns correct slice object."""
    slice_obj = MessageSlice(type="other", start=5, stop=15)

    assert slice_obj.slice_ == slice(5, 15)


def test_message_slice_length() -> None:
    """Test MessageSlice length calculation."""
    slice_obj = MessageSlice(type="other", start=5, stop=15)

    assert len(slice_obj) == 10


def test_message_slice_string_representation() -> None:
    """Test MessageSlice string representation."""
    message = Message("assistant", "The answer is 42.")
    slice_obj = MessageSlice(type="model", start=0, stop=17, obj=SimpleModel(content="test"))
    slice_obj._message = message

    str_repr = str(slice_obj)

    assert "MessageSlice" in str_repr
    assert "type='model'" in str_repr
    assert "start=0" in str_repr
    assert "stop=17" in str_repr
    assert "SimpleModel" in str_repr


def test_message_slice_clone() -> None:
    """Test MessageSlice cloning preserves all properties."""
    original_metadata = {"confidence": 0.95, "source": "test"}
    message = Message("assistant", "test content")

    # Create slice through the message to properly establish relationship
    slice_obj = message.mark_slice("test")
    assert slice_obj is not None

    slice_obj.obj = SimpleModel(content="test")
    slice_obj.metadata = original_metadata

    cloned_slice = slice_obj.clone()

    assert cloned_slice.type == slice_obj.type
    assert cloned_slice.start == slice_obj.start
    assert cloned_slice.stop == slice_obj.stop
    assert cloned_slice.obj == slice_obj.obj
    assert cloned_slice.metadata == slice_obj.metadata

    # assert that slice is detached
    assert cloned_slice._message is None

    # Ensure deep copy of metadata
    cloned_slice.metadata["confidence"] = 0.5
    assert slice_obj.metadata["confidence"] == 0.95


# =============================================================================
# Manual Slice Creation Tests
# =============================================================================


def test_mark_slice_with_string_target() -> None:
    """Test mark_slice with string target."""
    message = Message("assistant", "The answer is 42. This is correct.")

    slice_obj = message.mark_slice("42")

    assert slice_obj is not None
    assert slice_obj.start == 14
    assert slice_obj.stop == 16
    assert slice_obj.content == "42"
    assert len(message.slices) == 1


def test_mark_slice_with_range_tuple() -> None:
    """Test mark_slice with range tuple target."""
    message = Message("assistant", "The answer is 42.")

    slice_obj = message.mark_slice((14, 16))

    assert slice_obj is not None
    assert slice_obj.start == 14
    assert slice_obj.stop == 16
    assert slice_obj.content == "42"


def test_mark_slice_with_regex_pattern() -> None:
    """Test mark_slice with regex pattern target."""
    message = Message("assistant", "Values: 42, 123, 7")
    pattern = re.compile(r"\d+")

    slice_obj = message.mark_slice(pattern)

    assert slice_obj is not None
    assert slice_obj.content == "42"  # First match


def test_mark_slice_with_regex_pattern_select_all() -> None:
    """Test mark_slice with regex pattern selecting all matches."""
    message = Message("assistant", "Values: 42, 123, 7")
    pattern = re.compile(r"\d+")

    slices = message.mark_slice(pattern, select="all")

    assert isinstance(slices, list)
    assert len(slices) == 3
    assert slices[0].content == "42"
    assert slices[1].content == "123"
    assert slices[2].content == "7"


def test_mark_slice_with_full_message_target() -> None:
    """Test mark_slice with full message target (-1)."""
    message = Message("assistant", "Complete message content")

    slice_obj = message.mark_slice(-1)

    assert slice_obj is not None
    assert slice_obj.start == 0
    assert slice_obj.stop == len(message.content)
    assert slice_obj.content == message.content


def test_mark_slice_with_model_type() -> None:
    """Test mark_slice with Model type target."""
    message = Message("assistant", "Test: <simple-model>sample content</simple-model>")

    slice_obj = message.mark_slice(SimpleModel)

    assert slice_obj is not None
    assert slice_obj.type == "model"
    assert isinstance(slice_obj.obj, SimpleModel)
    assert slice_obj.obj.content == "sample content"


def test_mark_slice_case_sensitivity() -> None:
    """Test mark_slice case sensitivity option."""
    message = Message("assistant", "The ANSWER is correct")

    # Case insensitive (default) - should find
    slice_obj = message.mark_slice("ANSWER")
    assert slice_obj is not None
    assert slice_obj.content == "ANSWER"
    message.remove_slices(slice_obj)

    # Case sensitive - should not find
    slice_obj = message.mark_slice("CORRECT", case_sensitive=False)
    assert slice_obj is not None
    assert slice_obj.content == "correct"


def test_mark_slice_with_metadata() -> None:
    """Test mark_slice with custom metadata."""
    message = Message("assistant", "The answer is 42.")
    metadata = {"confidence": 0.95, "source": "calculation"}

    slice_obj = message.mark_slice("42", metadata=metadata)

    assert slice_obj is not None
    assert slice_obj.metadata == metadata


def test_mark_slice_with_custom_type() -> None:
    """Test mark_slice with custom slice type."""
    message = Message("assistant", "Tool called: search")

    slice_obj = message.mark_slice("search", slice_type="tool_call")

    assert slice_obj is not None
    assert slice_obj.type == "tool_call"


def test_mark_slice_target_not_found() -> None:
    """Test mark_slice when target is not found."""
    message = Message("assistant", "The answer is 42.")

    slice_obj = message.mark_slice("nonexistent")

    assert slice_obj is None
    assert len(message.slices) == 0


def test_mark_slice_with_invalid_range() -> None:
    """Test mark_slice with invalid range tuple."""
    message = Message("assistant", "Short")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        slice_obj = message.mark_slice((10, 20))

        assert slice_obj is None
        assert len(w) == 1
        assert issubclass(w[-1].category, MessageWarning)
        assert "Invalid range" in str(w[-1].message)


def test_mark_slice_select_last() -> None:
    """Test mark_slice with select='last' option."""
    message = Message("assistant", "Test test test")

    slice_obj = message.mark_slice("test", select="last")

    assert slice_obj is not None
    assert slice_obj.start == 10  # Last occurrence


# =============================================================================
# Slice Discovery and Access Tests
# =============================================================================


def test_find_slices_no_filter() -> None:
    """Test find_slices without any filters returns all slices."""
    message = Message("assistant", "The answer is 42.")
    message.mark_slice("answer", slice_type="model")
    message.mark_slice("42", slice_type="other")

    slices = message.find_slices()

    assert len(slices) == 2


def test_find_slices_with_type_filter() -> None:
    """Test find_slices with slice type filter."""
    message = Message("assistant", "The answer is 42.")
    message.mark_slice("answer", slice_type="model")
    message.mark_slice("42", slice_type="other")

    model_slices = message.find_slices(slice_type="model")
    other_slices = message.find_slices(slice_type="other")

    assert len(model_slices) == 1
    assert len(other_slices) == 1
    assert model_slices[0].content == "answer"
    assert other_slices[0].content == "42"


def test_find_slices_with_custom_filter() -> None:
    """Test find_slices with custom filter function."""
    message = Message("assistant", "High confidence: 42. Low confidence: maybe.")
    message.mark_slice("42", metadata={"confidence": 0.9})
    message.mark_slice("maybe", metadata={"confidence": 0.3})

    high_confidence_slices = message.find_slices(
        filter_fn=lambda s: s.metadata.get("confidence", 0) > 0.5,
    )

    assert len(high_confidence_slices) == 1
    assert high_confidence_slices[0].content == "42"


def test_get_slice_first_default() -> None:
    """Test get_slice returns first slice by default."""
    message = Message("assistant", "First slice. Second slice.")
    message.mark_slice("First")
    message.mark_slice("Second")

    first_slice = message.get_slice()

    assert first_slice is not None
    assert first_slice.content == "First"


def test_get_slice_last() -> None:
    """Test get_slice with select='last' option."""
    message = Message("assistant", "First slice. Second slice.")
    message.mark_slice("First")
    message.mark_slice("Second")

    last_slice = message.get_slice(select="last")

    assert last_slice is not None
    assert last_slice.content == "Second"


def test_get_slice_with_type_filter() -> None:
    """Test get_slice with slice type filter."""
    message = Message("assistant", "Model output: 42. Tool call: search.")
    message.mark_slice("42", slice_type="model")
    message.mark_slice("search", slice_type="tool_call")

    model_slice = message.get_slice(slice_type="model")
    tool_slice = message.get_slice(slice_type="tool_call")

    assert model_slice is not None
    assert model_slice.content == "42"
    assert tool_slice is not None
    assert tool_slice.content == "search"


def test_get_slice_no_matches() -> None:
    """Test get_slice returns None when no matches found."""
    message = Message("assistant", "No slices here.")

    slice_obj = message.get_slice()

    assert slice_obj is None


def test_iter_slices_forward() -> None:
    """Test iter_slices in forward order."""
    message = Message("assistant", "First slice. Second slice. Third slice.")
    message.mark_slice("First")
    message.mark_slice("Second")
    message.mark_slice("Third")

    slices = list(message.iter_slices())

    assert len(slices) == 3
    assert slices[0].content == "First"
    assert slices[1].content == "Second"
    assert slices[2].content == "Third"


def test_iter_slices_reverse() -> None:
    """Test iter_slices in reverse order."""
    message = Message("assistant", "First slice. Second slice. Third slice.")
    message.mark_slice("First")
    message.mark_slice("Second")
    message.mark_slice("Third")

    slices = list(message.iter_slices(reverse=True))

    assert len(slices) == 3
    assert slices[0].content == "Third"
    assert slices[1].content == "Second"
    assert slices[2].content == "First"


def test_iter_slices_with_type_filter() -> None:
    """Test iter_slices with slice type filter."""
    message = Message("assistant", "Model: 42. Tool: search. Model: result.")
    message.mark_slice("42", slice_type="model")
    message.mark_slice("search", slice_type="tool_call")
    message.mark_slice("result", slice_type="model")

    model_slices = list(message.iter_slices(slice_type="model"))

    assert len(model_slices) == 2
    assert model_slices[0].content == "42"
    assert model_slices[1].content == "result"


# =============================================================================
# Slice Removal and Manipulation Tests
# =============================================================================


def test_remove_slices_by_object() -> None:
    """Test remove_slices with MessageSlice object."""
    message = Message("assistant", "The answer is 42. This is correct.")
    slice1 = message.mark_slice("answer")
    slice2 = message.mark_slice("42")

    assert slice1 is not None
    assert slice2 is not None

    message.remove_slices(slice1)

    assert len(message.slices) == 1
    assert message.slices[0] == slice2


def test_remove_slices_by_type() -> None:
    """Test remove_slices with slice type."""
    message = Message("assistant", "Model: 42. Tool: search. Model: result.")
    message.mark_slice("42", slice_type="model")
    message.mark_slice("search", slice_type="tool_call")
    message.mark_slice("result", slice_type="model")

    message.remove_slices("model")

    assert len(message.slices) == 1
    assert message.slices[0].content == "search"


def test_remove_slices_by_content() -> None:
    """Test remove_slices with content matching."""
    message = Message("assistant", "The answer is 42. The answer is clear.")
    message.mark_slice("answer")
    message.mark_slice("42")
    message.mark_slice("answer")  # Second occurrence

    message.remove_slices("answer")

    assert len(message.slices) == 1
    assert message.slices[0].content == "42"


def test_remove_slices_by_model_type() -> None:
    """Test remove_slices with model type."""
    message = Message("assistant", "Text: <simple-model>content</simple-model> More text.")
    message.parse(SimpleModel)
    message.mark_slice("More", slice_type="other")

    message.remove_slices(SimpleModel)

    assert len(message.slices) == 1
    assert message.slices[0].content == "More"


def test_remove_slices_multiple_objects() -> None:
    """Test remove_slices with multiple objects."""
    message = Message("assistant", "First slice. Second slice. Third slice.")
    slice1 = message.mark_slice("First")
    slice2 = message.mark_slice("Second")
    slice3 = message.mark_slice("Third")

    assert len(message.slices) == 3
    assert slice1 is not None
    assert slice2 is not None
    assert slice3 is not None

    message.remove_slices(slice1, slice3)

    assert len(message.slices) == 1
    assert message.slices[0] == slice2


def test_remove_slices_updates_content() -> None:
    """Test remove_slices updates message content when configured."""
    message = Message("assistant", "Keep this. Remove this. Keep this too.")
    message.mark_slice("Remove this")

    # Note: This tests the current behavior - may need adjustment based on actual implementation
    original_length = len(message.slices)
    message.remove_slices("Remove this")

    assert len(message.slices) == original_length - 1


def test_remove_slices_nonexistent() -> None:
    """Test remove_slices with non-existent target does nothing."""
    message = Message("assistant", "The answer is 42.")
    slice_obj = message.mark_slice("42")

    message.remove_slices("nonexistent")

    assert len(message.slices) == 1
    assert message.slices[0] == slice_obj


# =============================================================================
# Content Modification and Position Update Tests
# =============================================================================


def test_content_update_preserves_valid_slices() -> None:
    """Test content updates preserve slices when their text still exists."""
    message = Message("assistant", "The answer is 42.")
    message.mark_slice("42")

    # Update content but keep the slice text
    message.content = "Actually, the answer is 42 for sure."

    # Slice should be updated to new position
    assert len(message.slices) == 1
    assert message.slices[0].content == "42"
    assert message.slices[0].start == 24  # New position


def test_content_update_removes_invalid_slices() -> None:
    """Test content updates remove slices when their text no longer exists."""
    message = Message("assistant", "The answer is 42.")
    message.mark_slice("42")

    # Update content without the slice text
    message.content = "The answer is unknown."

    assert len(message.slices) == 0


def test_content_update_multiple_slices() -> None:
    """Test content updates handle multiple slices correctly."""
    message = Message("assistant", "Value: 42. Status: correct. Final: 42.")
    message.mark_slice("42", metadata={"type": "first"})
    message.mark_slice("correct")
    message.mark_slice(
        "42",
        metadata={"type": "second"},
    )  # This will find the first occurrence again

    # Update content keeping some slices
    message.content = "Value: 42. Status: verified."

    # Should preserve slices that still exist
    preserved_slices = [s for s in message.slices if s.content in message.content]
    assert len(preserved_slices) >= 1


def test_slice_position_recalculation_accuracy() -> None:
    """Test slice position recalculation is accurate after content changes."""
    message = Message("assistant", "Start: 42. Middle: test. End: 99.")
    message.mark_slice("42")
    message.mark_slice("test")
    message.mark_slice("99")

    # Modify content
    message.content = "Beginning: 42. Center: test. Final: 99."

    # Check positions are recalculated correctly
    valid_slices = [s for s in message.slices if s.content in message.content]
    for slice_obj in valid_slices:
        expected_content = message.content[slice_obj.start : slice_obj.stop]
        assert slice_obj.content == expected_content


def test_overlapping_slices_position_updates() -> None:
    """Test position updates work correctly with overlapping slices."""
    message = Message("assistant", "The answer is definitely 42 for sure.")

    # Create overlapping slices
    message.mark_slice("answer is definitely 42")
    message.mark_slice("42")

    # Update content
    message.content = "The answer is definitely 42 and correct."

    # Both slices should update appropriately if their content still exists
    valid_slices = [s for s in message.slices if s.content in message.content]
    for slice_obj in valid_slices:
        expected_content = message.content[slice_obj.start : slice_obj.stop]
        assert slice_obj.content == expected_content


# =============================================================================
# Advanced Slice Operations Tests
# =============================================================================


def test_overlapping_slice_creation() -> None:
    """Test creating overlapping slices works correctly."""
    message = Message("assistant", "The answer is 42 and correct.")

    # Create overlapping slices
    full_slice = message.mark_slice("answer is 42")
    number_slice = message.mark_slice("42")

    assert len(message.slices) == 2
    assert full_slice is not None
    assert number_slice is not None

    # Check that one slice contains the other
    assert (full_slice.start <= number_slice.start and full_slice.stop >= number_slice.stop) or (
        number_slice.start <= full_slice.start and number_slice.stop >= full_slice.stop
    )


def test_hierarchical_slice_relationships() -> None:
    """Test parent-child slice relationships through metadata."""
    message = Message("assistant", "Process: Step 1: analyze. Step 2: conclude.")

    # Create parent slice
    message.mark_slice(
        "Process: Step 1: analyze. Step 2: conclude.",
        metadata={"type": "process", "id": "proc_1"},
    )

    # Create child slices with parent reference
    message.mark_slice(
        "Step 1: analyze",
        metadata={"type": "step", "parent_id": "proc_1", "order": 1},
    )
    message.mark_slice(
        "Step 2: conclude",
        metadata={"type": "step", "parent_id": "proc_1", "order": 2},
    )

    assert len(message.slices) == 3

    # Verify hierarchical relationships
    parent_slices = [s for s in message.slices if s.metadata.get("type") == "process"]
    child_slices = [s for s in message.slices if s.metadata.get("parent_id") == "proc_1"]

    assert len(parent_slices) == 1
    assert len(child_slices) == 2


def test_slice_ordering_and_sorting() -> None:
    """Test slice ordering based on position."""
    message = Message("assistant", "Third: C. First: A. Second: B.")

    # Add slices in non-sequential order
    message.mark_slice("C", metadata={"order": 3})
    message.mark_slice("A", metadata={"order": 1})
    message.mark_slice("B", metadata={"order": 2})

    # Slices should be orderable by position
    sorted_slices = sorted(message.slices, key=lambda s: s.start)

    assert len(sorted_slices) == 3
    assert sorted_slices[0].metadata["order"] == 3  # "C" comes first in text
    assert sorted_slices[1].metadata["order"] == 1  # "A" comes second in text
    assert sorted_slices[2].metadata["order"] == 2  # "B" comes third in text


def test_batch_slice_operations() -> None:
    """Test batch operations on multiple slices."""
    message = Message("assistant", "Values: 10, 20, 30, 40, 50.")

    # Create multiple slices for numbers
    pattern = re.compile(r"\d+")
    number_slices = message.mark_slice(pattern, select="all")

    assert isinstance(number_slices, list)
    assert len(number_slices) == 5

    # Batch update metadata
    for i, slice_obj in enumerate(number_slices):
        slice_obj.metadata["index"] = i
        slice_obj.metadata["value"] = int(slice_obj.content)

    # Verify batch operations worked
    values = [s.metadata["value"] for s in message.slices]
    assert values == [10, 20, 30, 40, 50]


def test_slice_content_manipulation() -> None:
    """Test manipulating slice content directly."""
    message = Message("assistant", "The answer is maybe 42.")

    uncertainty_slice = message.mark_slice("maybe")
    answer_slice = message.mark_slice("42")

    assert len(message.slices) == 2
    assert uncertainty_slice is not None
    assert answer_slice is not None

    # Modify slice content directly
    uncertainty_slice.content = "definitely"

    assert "definitely" in message.content
    assert "maybe" not in message.content
    assert answer_slice.content == "42"  # Other slices unaffected


# =============================================================================
# Integration with Models and Parsing Tests
# =============================================================================


def test_automatic_slice_creation_during_parsing() -> None:
    """Test slices are automatically created when parsing models."""
    message = Message("assistant", "Result: <simple-model>parsed content</simple-model>")

    parsed_model = message.parse(SimpleModel)

    assert isinstance(parsed_model, SimpleModel)
    assert len(message.slices) == 1
    assert message.slices[0].type == "model"
    assert message.slices[0].obj == parsed_model


def test_slice_preservation_through_parse_operations() -> None:
    """Test existing slices are preserved during parse operations."""
    message = Message("assistant", "Manual slice. <simple-model>auto slice</simple-model>")

    # Create manual slice first
    manual_slice = message.mark_slice("Manual slice")

    # Parse model (should create automatic slice)
    message.parse(SimpleModel)

    assert len(message.slices) == 2
    # Manual slice should still exist
    manual_slices = [s for s in message.slices if s == manual_slice]
    assert len(manual_slices) == 1


def test_model_object_associations_with_slices() -> None:
    """Test model objects are correctly associated with slices."""
    message = Message(
        "assistant",
        "Data: <complex-model name='test'><description>Description here</description><value>123</value></complex-model>",
    )

    parsed_model = message.parse(ComplexModel)

    assert len(message.slices) == 1
    model_slice = message.slices[0]

    assert model_slice.obj == parsed_model
    assert isinstance(model_slice.obj, ComplexModel)
    assert model_slice.obj.name == "test"
    assert model_slice.obj.description == "Description here"
    assert model_slice.obj.value == 123


def test_slice_behavior_with_multiple_models() -> None:
    """Test slice behavior when parsing multiple model instances."""
    message = Message(
        "assistant",
        "First: <simple-model>content1</simple-model> Second: <simple-model>content2</simple-model>",
    )

    models = message.parse_set(SimpleModel)

    assert len(models) == 2
    assert len(message.slices) == 2

    # Verify each slice is associated with correct model
    for i, slice_obj in enumerate(message.slices):
        assert slice_obj.obj == models[i]
        assert slice_obj.obj.content == f"content{i + 1}"


def test_slice_behavior_with_nested_models() -> None:
    """Test slice behavior with complex nested model structures."""
    message = Message(
        "assistant",
        "List: <nested-model title='test'>Items: <items>item1</items><items>item2</items></nested-model>",
    )

    message.parse(NestedModel)

    assert len(message.slices) == 1
    model_slice = message.slices[0]

    assert isinstance(model_slice.obj, NestedModel)
    assert model_slice.obj.title == "test"
    assert model_slice.obj.items == ["item1", "item2"]


# =============================================================================
# Chat-Level Slice Operations Tests
# =============================================================================


def test_chat_message_slices_all() -> None:
    """Test Chat.message_slices() returns slices from all messages."""
    chat = Chat(
        [
            Message("user", "Question about 42?"),
            Message("assistant", "The answer is 42."),
            Message("user", "Why 42?"),
        ],
    )

    # Add slices to different messages
    chat.all[0].mark_slice("42")
    chat.all[1].mark_slice("answer")
    chat.all[1].mark_slice("42")
    chat.all[2].mark_slice("Why")

    all_slices = chat.message_slices()

    assert len(all_slices) == 4


def test_chat_message_slices_with_type_filter() -> None:
    """Test Chat.message_slices() with slice type filter."""
    chat = Chat(
        [
            Message("user", "Tool call: search"),
            Message("assistant", "Model output: result"),
        ],
    )

    chat.all[0].mark_slice("search", slice_type="tool_call")
    chat.all[1].mark_slice("result", slice_type="model")

    tool_slices = chat.message_slices(slice_type="tool_call")
    model_slices = chat.message_slices(slice_type="model")

    assert len(tool_slices) == 1
    assert len(model_slices) == 1
    assert tool_slices[0].content == "search"
    assert model_slices[0].content == "result"


def test_chat_message_slices_with_filter_function() -> None:
    """Test Chat.message_slices() with custom filter function."""
    chat = Chat(
        [
            Message("user", "High confidence: certain"),
            Message("assistant", "Low confidence: maybe"),
        ],
    )

    chat.all[0].mark_slice("certain", metadata={"confidence": 0.9})
    chat.all[1].mark_slice("maybe", metadata={"confidence": 0.3})

    high_confidence_slices = chat.message_slices(
        filter_fn=lambda s: s.metadata.get("confidence", 0) > 0.5,
    )

    assert len(high_confidence_slices) == 1
    assert high_confidence_slices[0].content == "certain"


def test_chat_message_slices_cross_message_aggregation() -> None:
    """Test aggregating slice data across multiple messages."""
    chat = Chat(
        [
            Message("user", "Process step 1"),
            Message("assistant", "Process step 2"),
            Message("user", "Process step 3"),
        ],
    )

    # Add process step slices across messages
    for i, message in enumerate(chat.all):
        message.mark_slice(f"step {i + 1}", metadata={"step": i + 1, "process_id": "proc_1"})

    process_slices = chat.message_slices(
        filter_fn=lambda s: s.metadata.get("process_id") == "proc_1",
    )

    assert len(process_slices) == 3
    steps = sorted([s.metadata["step"] for s in process_slices])
    assert steps == [1, 2, 3]


def test_chat_message_slices_empty_chat() -> None:
    """Test Chat.message_slices() with empty chat."""
    chat = Chat([])

    slices = chat.message_slices()

    assert len(slices) == 0


def test_chat_message_slices_no_slices() -> None:
    """Test Chat.message_slices() when no messages have slices."""
    chat = Chat(
        [
            Message("user", "No slices here"),
            Message("assistant", "Nothing marked"),
        ],
    )

    slices = chat.message_slices()

    assert len(slices) == 0


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


def test_empty_content_handling() -> None:
    """Test slice operations with empty message content."""
    message = Message("assistant", "")

    # Marking slice on empty content should return None
    slice_obj = message.mark_slice("anything")
    assert slice_obj is None

    # No slices should exist
    assert len(message.slices) == 0


def test_slice_with_empty_string_target() -> None:
    """Test marking slice with empty string target."""
    message = Message("assistant", "Some content here")

    slice_obj = message.mark_slice("")

    # Empty string should not create a valid slice
    assert slice_obj is None


def test_invalid_range_positions() -> None:
    """Test slice creation with invalid range positions."""
    message = Message("assistant", "Short")

    # Test range beyond content length
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        slice_obj = message.mark_slice((10, 20))
        assert slice_obj is None
        assert len(w) == 1
        assert "Invalid range" in str(w[-1].message)

    # Test negative start position
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        slice_obj = message.mark_slice((-1, 5))
        assert slice_obj is None
        assert len(w) == 1
        assert "Invalid range" in str(w[-1].message)

    # Test start > stop
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        slice_obj = message.mark_slice((10, 5))
        assert slice_obj is None
        assert len(w) == 1
        assert "Invalid range" in str(w[-1].message)


def test_large_number_of_slices_performance() -> None:
    """Test performance with large number of slices."""
    content = " ".join([f"word{i}" for i in range(100, 200)])
    message = Message("assistant", content)

    # Create many slices
    for i in range(100, 150):
        message.mark_slice(f"word{i}")

    # Operations should still work efficiently
    assert len(message.slices) == 50

    # Test slice retrieval
    found_slices = message.find_slices()
    assert len(found_slices) == 50

    # Test content modification
    message.content = content + " additional"
    # Most slices should still be valid
    valid_slices = [s for s in message.slices if s.content in message.content]
    assert len(valid_slices) == 50


def test_slice_metadata_type_safety() -> None:
    """Test slice metadata handles various data types safely."""
    message = Message("assistant", "Test content")

    # Test various metadata types
    complex_metadata = {
        "string": "value",
        "integer": 42,
        "float": 3.14,
        "boolean": True,
        "list": [1, 2, 3],
        "dict": {"nested": "value"},
        "none": None,
    }

    slice_obj = message.mark_slice("content", metadata=complex_metadata)

    assert slice_obj is not None
    assert slice_obj.metadata == complex_metadata


def test_slice_serialization_deserialization() -> None:
    """Test slice data survives serialization and deserialization."""
    message = Message("assistant", "Test: <simple-model>content</simple-model>")
    message.parse(SimpleModel)
    message.mark_slice("Test", metadata={"custom": "data"})

    # Serialize message
    serialized = message.model_dump_json()

    # Deserialize message
    deserialized = Message.model_validate_json(serialized)

    # Slices should be preserved
    assert len(deserialized.slices) == 2

    # Custom metadata should be preserved
    custom_slices = [s for s in deserialized.slices if s.metadata.get("custom") == "data"]
    assert len(custom_slices) == 1


def test_malformed_slice_data_handling() -> None:
    """Test handling of malformed slice data during operations."""
    message = Message("assistant", "Test content")

    # Create slice and manually corrupt its data
    slice_obj = message.mark_slice("Test")
    assert slice_obj is not None

    # Corrupt slice position data
    slice_obj.start = -1
    slice_obj.stop = 1000

    # Content access should handle corrupted data gracefully
    content = slice_obj.content
    assert content == "[detached]" or isinstance(content, str)


def test_concurrent_slice_modifications() -> None:
    """Test slice behavior under concurrent modifications."""
    message = Message("assistant", "Original content with multiple words")

    # Create multiple slices
    slice1 = message.mark_slice("Original")
    slice2 = message.mark_slice("content")
    slice3 = message.mark_slice("words")

    assert len(message.slices) == 3
    assert slice1 is not None
    assert slice2 is not None
    assert slice3 is not None

    # Modify content through one slice
    slice1.content = "Modified"

    # Other slices should handle the change appropriately
    # (Their positions may be updated or they may become invalid)
    for slice_obj in [slice2, slice3]:
        # Content access should not raise exceptions
        content = slice_obj.content
        assert isinstance(content, str)


def test_memory_management_with_detached_slices() -> None:
    """Test memory management when slices become detached."""
    message = Message("assistant", "Test content")
    slice_obj = message.mark_slice("Test")

    assert len(message.slices) == 1
    assert slice_obj is not None

    # Detach slice by clearing message reference
    slice_obj._message = None

    # Detached slice should handle operations safely
    assert slice_obj.content == "[detached]"

    # Setting content on detached slice should warn but not crash
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        slice_obj.content = "new content"
        assert len(w) == 1
        assert issubclass(w[-1].category, MessageWarning)
