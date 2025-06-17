import typing as t
import rigging as rg
import dreadnode as dn


def chat_accuracy(
    values: list[rg.Model],
    parse_many: bool = False,
) -> t.Callable[[rg.Chat], rg.Chat]:
    """
    Measures accuracy between the `values` parsed outputs extracted from `chat.last.try_parse_many()`.

    - Comparison is element-wise up to the length of the longer list (`max_len`).
    - Missing items (ground truth vs. reported) or extra items are penalized as
      denominators for accuracy calculations use `max_len`.
    - A field is correct if `ans_value == rep_value` (this includes `None == None`).
    - `overall_items` (accuracy): Percentage of item *pairs* (value_item, report_item)
      at an index `i` where both items exist and all their `fields_to_check` values are equal.
    - `overall_fields` (accuracy): Percentage of total individual field "slots" that match
      across all compared item pairs up to `max_len`.
    - Per-field accuracy: Accuracy for each field in `fields_to_check`.

    """

    fields_to_check: list[str] = []
    if values:  # Derives fields from the first ground truth item
        fields_to_check = list(values[0].model_fields.keys())

    num_fields_per_item = len(fields_to_check)

    if num_fields_per_item == 0:
        raise ValueError(
            "No fields to check. Ensure that `values` contains items with model fields."
        )

    def evaluate(
        chat: rg.Chat,
    ) -> rg.Chat:
        if parse_many:
            outputs_raw = chat.last.try_parse_many()
        else:
            outputs_raw = chat.last.try_parse()
            outputs_raw = [outputs_raw]

        outputs = outputs_raw if isinstance(outputs_raw, list) else []

        total_value_items = len(values)
        total_reported_items = len(outputs)

        max_len = max(
            total_value_items, total_reported_items
        )  # Length of the longer list

        if max_len == 0:  # Both lists are empty
            for field_name in fields_to_check:
                dn.log_metric(field_name, 0.0)

            return chat

        # Initialize counters for correct matches
        correct_field_counts: dict[str, int] = dict.fromkeys(fields_to_check, 0)
        matching_overall_items_count = 0
        total_correct_individual_fields = 0

        for i in range(max_len):
            value_item = values[i] if i < total_value_items else None
            report_item = outputs[i] if i < total_reported_items else None

            current_item_all_fields_match = (
                False  # Assume false unless proven otherwise
            )
            if (
                value_item and report_item
            ):  # Both items must exist for an overall item match
                all_fields_in_item_pair_identical = True
                if (
                    not fields_to_check
                ):  # If no fields to check (e.g. EmptyModel), they are identical
                    pass
                else:
                    for field_name in fields_to_check:
                        value_attr = getattr(value_item, field_name, None)
                        report_attr = getattr(report_item, field_name, None)
                        if value_attr != report_attr:
                            all_fields_in_item_pair_identical = False
                            break
                if all_fields_in_item_pair_identical:
                    current_item_all_fields_match = True

            if current_item_all_fields_match:
                matching_overall_items_count += 1

            # Score individual fields for the current item pair (value_item, report_item)
            # This contributes to partial credit.
            for field_name in fields_to_check:
                value_attr = getattr(value_item, field_name, None)
                report_attr = getattr(report_item, field_name, None)

                if (
                    value_attr == report_attr
                ):  # `None == None` is considered a match here
                    correct_field_counts[field_name] += 1
                    total_correct_individual_fields += 1

        # Calculate and store accuracies
        # Denominator is max_len, so missing/extra items penalize the score.
        for field_name in fields_to_check:
            field_accuracy = correct_field_counts[field_name] / max_len
            dn.log_metric(field_name, field_accuracy)

        total_possible_individual_fields = max_len * num_fields_per_item
        overall_fields_accuracy = (
            total_correct_individual_fields / total_possible_individual_fields
        )

        dn.log_metric("accuracy", overall_fields_accuracy)

        return chat

    return evaluate
