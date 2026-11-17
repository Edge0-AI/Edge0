use std::collections::BTreeSet;

/// Apply only the locations the page added or removed since its last snapshot.
/// A download can add a location while that snapshot waits in the save queue.
pub fn merge_asset_locations(
    current: &[String],
    previous: &[String],
    requested: &[String],
) -> Vec<String> {
    let previous: BTreeSet<_> = previous.iter().collect();
    let requested_set: BTreeSet<_> = requested.iter().collect();
    let mut seen = BTreeSet::new();
    current
        .iter()
        .filter(|path| !previous.contains(path) || requested_set.contains(path))
        .chain(requested.iter().filter(|path| !previous.contains(path)))
        .filter(|path| seen.insert(*path))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::merge_asset_locations;
    fn paths(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn keeps_a_download_finished_after_the_page_snapshot() {
        let previous = paths(&["/existing"]);
        let current = paths(&["/existing", "/downloaded"]);
        assert_eq!(
            merge_asset_locations(&current, &previous, &previous),
            current
        );
    }

    #[test]
    fn removes_only_the_explicitly_removed_location() {
        assert_eq!(
            merge_asset_locations(
                &paths(&["/old", "/keep", "/downloaded"]),
                &paths(&["/old", "/keep"]),
                &paths(&["/keep"])
            ),
            paths(&["/keep", "/downloaded"])
        );
    }

    #[test]
    fn does_not_restore_a_location_removed_by_an_earlier_save() {
        assert_eq!(
            merge_asset_locations(
                &paths(&["/second", "/downloaded"]),
                &paths(&["/first", "/second"]),
                &paths(&["/first"])
            ),
            paths(&["/downloaded"])
        );
    }

    #[test]
    fn adds_each_location_once_and_preserves_current_order() {
        assert_eq!(
            merge_asset_locations(
                &paths(&["/second", "/first", "/second"]),
                &paths(&["/first", "/second"]),
                &paths(&["/first", "/second", "/new", "/new"])
            ),
            paths(&["/second", "/first", "/new"])
        );
    }

    #[test]
    fn preserves_exact_native_paths_including_roots() {
        assert_eq!(
            merge_asset_locations(&paths(&["/"]), &[], &paths(&["Z:\\", "C:\\models\\", "/"])),
            paths(&["/", "Z:\\", "C:\\models\\"])
        );
    }

    #[test]
    fn an_explicit_removal_wins_for_a_location_already_known_to_the_page() {
        assert!(merge_asset_locations(&paths(&["/known"]), &paths(&["/known"]), &[]).is_empty());
    }
}
