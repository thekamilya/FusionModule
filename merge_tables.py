import pandas as pd


def merge_medical_data(labels_file, patient_files, output_file, scan_type):
    """
    Merges patient label data with patient file data for medical imaging.
    To bring to format:
    case_id,chosen_exam
    C3L-00004,C3L-00004-26.npz
    C3L-00010,C3L-00010-26.npz
    C3L-00011,C3L-00011-26.npz
    C3L-00026,C3L-00026-21.npz
    C3L-00079,C3L-00079-26.npz

    So that each case has exactly one imaging data.

    Parameters:
    -----------
    labels_file : str
        Path to CSV with patient labels (e.g., 'patients_with_labels_CT_final.csv')
    patient_files : str
        Path to CSV with patient files (e.g., 'CT_patientfiles.csv')
    output_file : str
        Path for output CSV (e.g., 'CT_FINAL.csv')
    scan_type : str
        Type of scan for logging purposes (e.g., 'CT' or 'MRI')

    Returns:
    --------
    pd.DataFrame
        Merged dataframe with case_id and chosen_exam columns
    """

    print(f"\n{'=' * 50}")
    print(f"Processing {scan_type} data")
    print(f"{'=' * 50}\n")

    # Load patient labels
    df_labels = pd.read_csv(labels_file)
    print(f"Loaded {labels_file}")
    print(f"Shape: {df_labels.shape}")
    print(df_labels.head())

    # Add numbering to duplicate case_ids and create new identifiers
    df_labels['new_case_id'] = df_labels.groupby('case_id').cumcount() + 1
    df_labels['case_id_with_number'] = (
            df_labels['case_id'] + '-' +
            df_labels['new_case_id'].astype(str) + '.npz'
    )
    df_labels.rename(columns={'File': 'chosen_exam'}, inplace=True)

    print("\nAfter adding case numbering:")
    print(df_labels[['case_id', 'new_case_id', 'case_id_with_number']].head())

    # Load patient files
    df_patients = pd.read_csv(patient_files)
    print(f"\nLoaded {patient_files}")
    print(f"Shape: {df_patients.shape}")

    # Merge datasets on chosen_exam column
    merged = pd.merge(df_labels, df_patients, on='chosen_exam')

    # Select and rename final columns
    final_data = merged[['case_id_x', 'case_id_with_number']].copy()
    final_data.rename(
        columns={
            'case_id_x': 'case_id',
            'case_id_with_number': 'chosen_exam'
        },
        inplace=True
    )

    print("\nFinal merged data:")
    print(f"Shape: {final_data.shape}")
    print(final_data.head())

    # Save to CSV
    final_data.to_csv(output_file, index=False)
    print(f"\n✓ Saved to '{output_file}'")

    return final_data


if __name__ == "__main__":
    # Process CT data
    ct_data = merge_medical_data(
        labels_file="mmist_data/patients_with_labels_CT_final.csv",
        patient_files="mmist_data/CT_patientfiles.csv",
        output_file="mmist_data/CT_Merged.csv",
        scan_type="CT"
    )

    # Process MRI data
    mri_data = merge_medical_data(
        labels_file="mmist_data/patients_with_labels_MR_final.csv",
        patient_files="mmist_data/MRI_patientfiles.csv",
        output_file="mmist_data/MRI_Merged.csv",
        scan_type="MRI"
    )

    print("\n" + "=" * 50)
    print("All processing complete!")
    print("=" * 50)