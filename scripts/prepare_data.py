#!/usr/bin/env python
"""
This script processes crystal structure data and prepares 
training/validation/test datasets in LMDB format for machine learning models.

Usage examples:
    
    # data paths, target property, and split ratios
    python prepare_data.py --csv_file my_data.csv --target_property Formation_Energy
                           --split_ratios 0.8 0.1 0.1
    
    # Limit dataset size for testing or quick processing by -max_samples flag.
    python prepare_data.py ... --max_samples 1000 

    # Skip analysis for faster processing by --no_analysis flag.
    python prepare_data.py ... --no_analysis

    # Specify output directory for processed LMDB datasets
    python prepare_data.py ... --output_dir my_output_dir
"""

import argparse
import os
import shutil
import pickle

import numpy as np
import pandas as pd
import ast
import lmdb
import torch
import spglib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from ase.io import read
from pymatgen.core import Lattice, Structure
from fairchem.core.preprocessing import AtomsToGraphs
from fairchem.core.datasets import LmdbDataset



# Utility functions for data processing
def str_to_2d_array(string):
    """
    Convert string representation to numpy array.
    
    The object in pandas dataframe is string format.
    Convert to numpy array.
    
    Args:
        string (str): String representation of array
        
    Returns:
        np.ndarray or None: Converted array or None if conversion fails
    """
    if ',' not in string:
        string = string.replace(' ', ',')
    try:
        list_of_lists = ast.literal_eval(string)
        return np.array(list_of_lists)
    except ValueError:
        return None


def symmetrize_structure(structure, symprec=0.001):
    """
    Convert non-primitive to primitive cell using spglib.
    
    Args:
        structure: pymatgen Structure object
        symprec (float): Symmetry precision for spglib
        
    Returns:
        tuple: (primitive_structure, spacegroup_symbol)
    """
    cell = (structure.lattice.matrix, structure.frac_coords, structure.atomic_numbers)
    try:
        lattice, scaled_positions, numbers = spglib.standardize_cell(
            cell, 
            to_primitive=True, 
            no_idealize=False, 
            symprec=symprec
        )
        spacegroup_symbol = spglib.get_spacegroup(cell, symprec=symprec)
        return Structure(Lattice(lattice), numbers, scaled_positions), spacegroup_symbol
    except Exception:
        return structure, None


def get_structure(system):
    """
    Convert pandas Series row to pymatgen Structure object.
    
    Args:
        system (pd.Series): One compound row from dataframe
        
    Returns:
        Structure: pymatgen Structure object
    """
    cell = str_to_2d_array(system['cell'])
    posi = str_to_2d_array(system['positions'])    
    atom = str_to_2d_array(system['numbers'])
    
    lattice = Lattice(cell)
    structure = Structure(lattice, atom, posi)
    structure, _ = symmetrize_structure(structure)

    return structure


def split_dataset(atoms_list, ratio=[0.8, 0.1, 0.1], seed=None):
    """
    Split dataset into train/validation/test sets by given ratio.
    
    Args:
        atoms_list (list): List of ASE atoms objects
        ratio (list): Split ratios for [train, val, test]
        seed (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary with 'train', 'val', 'test' keys containing split data
    """
    N = len(atoms_list)
    ratio = np.array(ratio)
    assert ratio.sum() == 1.0, "Ratios must sum to 1.0"
    ratio /= ratio.sum()    

    train_end = int(N * ratio[0])
    val_end = train_end + int(N * ratio[1])
    
    ids = np.arange(N)
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(ids)
    
    split_set = {
        'train': [],
        'val': [],
        'test': []
    }
    
    for ind in ids[0:train_end]:
        split_set['train'].append(atoms_list[ind])
    for ind in ids[train_end:val_end]:
        split_set['val'].append(atoms_list[ind])
    for ind in ids[val_end:]:
        split_set['test'].append(atoms_list[ind])

    return split_set


def split_set_to_lmdb(split_set, properties, dir_name):
    """
    Convert split dataset to LMDB database format for training.
    
    Args:
        split_set (dict): Dictionary with train/val/test splits
        properties (dict): Property names mapping
        dir_name (str): Output directory name
    """
    # Setup atoms to graph converter
    a2g = AtomsToGraphs(
        r_energy=False,    
        r_forces=False,    
        r_distances=True,
        r_edges=True,
        r_fixed=True,
        r_pbc=True,
        r_data_keys=['id'] + list(properties.keys())
    )

    # Remove existing directory and create new one
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)

    # Create LMDB database for each split
    for db_name in split_set.keys():
        raw_data = split_set[db_name]
        
        print(f"Processing {db_name} set with {len(raw_data)} samples...")
    
        db = lmdb.open(
            f"{dir_name}/{db_name}.lmdb",
            map_size=1099511627776 * 2,
            subdir=False,
            meminit=False,
            map_async=True,
        )
        
        # Validate raw data before conversion
        validated_data = []
        skipped_count = 0
        
        for i, atoms in enumerate(raw_data):
            try:
                # Check if atoms object is valid
                if atoms is None:
                    print(f"Warning: Skipping None atoms object at index {i}")
                    skipped_count += 1
                    continue
                
                # Check if atoms has positions
                if not hasattr(atoms, 'positions') or len(atoms.positions) == 0:
                    print(f"Warning: Skipping atoms with no positions at index {i}")
                    skipped_count += 1
                    continue
                
                # Check if required properties exist
                for prop_key in properties.keys():
                    if prop_key not in atoms.info:
                        print(f"Warning: Missing property '{prop_key}' in atoms at index {i}")
                        skipped_count += 1
                        continue
                
                # Check for NaN values in properties
                valid_props = True
                for prop_key in properties.keys():
                    prop_value = atoms.info[prop_key]
                    if prop_value is None or (isinstance(prop_value, (int, float)) and np.isnan(prop_value)):
                        print(f"Warning: Invalid property value '{prop_key}={prop_value}' at index {i}")
                        valid_props = False
                        break
                
                if not valid_props:
                    skipped_count += 1
                    continue
                
                validated_data.append(atoms)
                
            except Exception as e:
                print(f"Warning: Error validating atoms at index {i}: {e}")
                skipped_count += 1
                continue
        
        if skipped_count > 0:
            print(f"Skipped {skipped_count} invalid entries in {db_name} set")
        
        print(f"Converting {len(validated_data)} validated samples to graph objects...")
        
        # Convert validated data to graph objects
        try:
            data_objects = a2g.convert_all(validated_data, disable_tqdm=True)
        except Exception as e:
            print(f"Error during graph conversion for {db_name} set: {e}")
            db.close()
            continue
        
        # Validate converted data objects
        valid_data_objects = []
        conversion_skipped = 0
        
        for fid, data in tqdm(enumerate(data_objects), total=len(data_objects),
                              desc=f"Validating {db_name} set"):
            try:
                # Check if data object is valid
                if data is None:
                    print(f"Warning: None data object at index {fid}")
                    conversion_skipped += 1
                    continue
                
                # Check if data has required attributes
                if not hasattr(data, 'pos') or data.pos is None:
                    print(f"Warning: Missing positions in data object at index {fid}")
                    conversion_skipped += 1
                    continue
                
                if not hasattr(data, 'edge_index'):
                    print(f"Warning: Missing edge_index in data object at index {fid}")
                    conversion_skipped += 1
                    continue
                
                # Check for edge connectivity (skip isolated atoms)
                if data.edge_index.shape[1] == 0:
                    print(f"Warning: No neighbors found for sample {fid}, skipping")
                    conversion_skipped += 1
                    continue
                
                # Add edge distance vectors if missing
                if not hasattr(data, 'edge_distance_vec'):
                    try:
                        # Calculate edge vectors
                        row, col = data.edge_index
                        edge_vec = data.pos[row] - data.pos[col]
                        
                        # Handle periodic boundary conditions if available
                        if hasattr(data, 'cell') and data.cell is not None:
                            # Apply minimum image convention
                            cell = data.cell.view(-1, 3)
                            # Simplified PBC handling
                            inv_cell = torch.linalg.pinv(cell)
                            frac_vec = edge_vec @ inv_cell
                            frac_vec = frac_vec - torch.round(frac_vec)
                            edge_vec = frac_vec @ cell
                        
                        data.edge_distance_vec = edge_vec
                        
                        # Also ensure edge distances are available
                        if not hasattr(data, 'distances'):
                            data.distances = torch.norm(edge_vec, dim=1)
                            
                    except Exception as e:
                        print(f"Warning: Failed to calculate edge vectors "
                              f"for sample {fid}: {e}")
                        conversion_skipped += 1
                        continue
                
                # Assign sid and fid
                data.sid = torch.LongTensor([0])
                data.fid = torch.LongTensor([len(valid_data_objects)])
                
                valid_data_objects.append(data)
                
            except Exception as e:
                print(f"Warning: Error processing data object at index {fid}: {e}")
                conversion_skipped += 1
                continue
        
        if conversion_skipped > 0:
            print(f"Skipped {conversion_skipped} invalid data objects in {db_name} set")
        
        print(f"Writing {len(valid_data_objects)} valid entries to LMDB...")
        
        # Write valid data objects to LMDB
        for final_id, data in tqdm(enumerate(valid_data_objects), 
                                  total=len(valid_data_objects),
                                  desc=f"Writing {db_name} LMDB"):
            try:
                # Update fid to reflect final position
                data.fid = torch.LongTensor([final_id])
                
                # Serialize and store
                serialized_data = pickle.dumps(data, protocol=-1)
                if serialized_data is None:
                    print(f"Warning: Failed to serialize data at final_id {final_id}")
                    continue
                
                txn = db.begin(write=True)
                txn.put(f"{final_id}".encode("ascii"), serialized_data)
                txn.commit()
                
            except Exception as e:
                print(f"Error writing data to LMDB at final_id {final_id}: {e}")
                continue
        
        # Write length information
        try:
            txn = db.begin(write=True)
            txn.put(f"length".encode("ascii"), pickle.dumps(len(valid_data_objects), protocol=-1))
            txn.commit()
        except Exception as e:
            print(f"Error writing length to LMDB: {e}")
            
        db.sync()
        db.close()
        
        print(f"Successfully created {db_name}.lmdb with {len(valid_data_objects)} entries")


def validate_lmdb_database(lmdb_path, max_samples_to_check=10):
    """
    Validate LMDB database for potential issues.
    
    Args:
        lmdb_path (str): Path to LMDB database
        max_samples_to_check (int): Maximum number of samples to validate
        
    Returns:
        bool: True if database appears valid
    """
    try:
        from fairchem.core.datasets import LmdbDataset
        dataset = LmdbDataset({"src": lmdb_path})
        
        print(f"Validating LMDB database: {lmdb_path}")
        print(f"Dataset length: {len(dataset)}")
        
        # Check a few samples
        samples_to_check = min(max_samples_to_check, len(dataset))
        issues_found = 0
        
        for i in range(samples_to_check):
            try:
                data = dataset[i]
                if data is None:
                    print(f"Issue: Sample {i} is None")
                    issues_found += 1
                    continue
                    
                if not hasattr(data, 'pos'):
                    print(f"Issue: Sample {i} missing positions")
                    issues_found += 1
                    
                if not hasattr(data, 'edge_index'):
                    print(f"Issue: Sample {i} missing edge_index")
                    issues_found += 1
                    
            except Exception as e:
                print(f"Issue: Error reading sample {i}: {e}")
                issues_found += 1
        
        if issues_found == 0:
            print(f"✓ Database validation passed ({samples_to_check} samples checked)")
            return True
        else:
            print(f"✗ Database validation failed ({issues_found} issues found)")
            return False
            
    except Exception as e:
        print(f"Error validating database: {e}")
        return False


def db_to_atomslist(compounds_df, dir_poscar='structures_poscar', 
                    properties=None):
    """
    Convert pandas DataFrame to list of ASE atoms objects.
    
    Args:
        compounds_df (pd.DataFrame): DataFrame containing materials data
        dir_poscar (str): Directory containing POSCAR/VASP files
        properties (dict): Property names mapping
        
    Returns:
        list: List of ASE atoms objects with attached properties
    """
    if properties is None:
        properties = {'material_id': 'material_id'}
    
    atoms_list = []
    skipped_count = 0
    
    print(f"Processing {len(compounds_df)} compounds...")
    
    for i in tqdm(compounds_df.index, desc="Loading structures"):
        try:
            material_id = compounds_df.loc[i]['material_id']
            
            # Convert material_id to string if it's numeric
            material_id_str = str(material_id)
            
            # Construct structure file path
            if 'mp-' not in material_id_str:
                structure_file = os.path.join(dir_poscar, f'mp-{material_id_str}.vasp')
            else:
                structure_file = os.path.join(dir_poscar, f'{material_id_str}.vasp')
            
            # Check if structure file exists
            if not os.path.exists(structure_file):
                print(f"Warning: Structure file not found: {structure_file}")
                skipped_count += 1
                continue
            
            # Read structure
            try:
                atoms = read(structure_file)
            except Exception as e:
                print(f"Warning: Failed to read structure {structure_file}: {e}")
                skipped_count += 1
                continue
            
            # Validate structure
            if atoms is None:
                print(f"Warning: None atoms object from {structure_file}")
                skipped_count += 1
                continue
            
            if len(atoms) == 0:
                print(f"Warning: Empty atoms object from {structure_file}")
                skipped_count += 1
                continue
            
            # Check for valid positions
            if not hasattr(atoms, 'positions') or atoms.positions is None:
                print(f"Warning: Invalid positions in {structure_file}")
                skipped_count += 1
                continue
            
            # Validate properties
            valid_props = True
            for key, item in properties.items():
                if item not in compounds_df.columns:
                    print(f"Warning: Property '{item}' not found in DataFrame")
                    valid_props = False
                    break
                
                prop_value = compounds_df.loc[i][item]
                if prop_value is None:
                    print(f"Warning: None value for property '{item}' in row {i}")
                    valid_props = False
                    break
                
                # Check for NaN values in numeric properties
                if isinstance(prop_value, (int, float)) and np.isnan(prop_value):
                    print(f"Warning: NaN value for property '{item}' in row {i}")
                    valid_props = False
                    break
            
            if not valid_props:
                skipped_count += 1
                continue
            
            # Attach metadata to atoms object
            atoms.info['id'] = i
            for key, item in properties.items():
                atoms.info[key] = compounds_df.loc[i][item]
            
            atoms_list.append(atoms)
            
        except Exception as e:
            print(f"Warning: Error processing compound at index {i}: {e}")
            skipped_count += 1
            continue
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} invalid compounds")
    
    print(f"Successfully loaded {len(atoms_list)} valid structures")
    return atoms_list


def analyze_dataset_statistics(lmdb_path, target_prop, binwidth=0.1):
    """
    Analyze and visualize dataset statistics.
    
    Args:
        lmdb_path (str): Path to LMDB database
        target_prop (str): Target property name
        binwidth (float): Histogram bin width
    """
    try:
        dataset = LmdbDataset({"src": lmdb_path})
        values = [data[target_prop] for data in dataset]
        
        # Plot histogram
        plt.figure(figsize=(8, 6))
        sns.histplot(values, binwidth=binwidth)
        plt.title(f"Distribution of {target_prop}")
        plt.xlabel(target_prop)
        plt.ylabel("Count")
        plt.show()
        
        # Print statistics
        print(f"{'Mean:':<10}{np.mean(values):.4f}")
        print(f"{'Std Dev:':<10}{np.std(values):.4f}")
        print(f"{'Min:':<10}{np.min(values):.4f}")
        print(f"{'Max:':<10}{np.max(values):.4f}")
        print(f"{'Count:':<10}{len(values)}")
        
    except Exception as e:
        print(f"Error analyzing dataset: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare crystal structure data for ML training"
    )
    
    parser.add_argument(
        '--csv_file', 
        type=str,
        default='QSGW_dataset/tab_nonmag_all.csv',
        help='Path to CSV file containing material data'
    )
    
    parser.add_argument(
        '--poscar_dir',
        type=str, 
        default='QSGW_dataset/structures_poscar',
        help='Directory containing POSCAR/VASP structure files'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for LMDB databases (default: set_{target_property}_train)'
    )
    
    parser.add_argument(
        '--target_property',
        type=str,
        default='2shot',
        help='Target property name to predict'
    )
    
    parser.add_argument(
        '--split_ratios',
        nargs=3,
        type=float,
        default=[0.8, 0.15, 0.05],
        help='Split ratios for train/val/test (must sum to 1.0)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to use from CSV file (default: use all)'
    )
    
    parser.add_argument(
        '--no_analysis',
        action='store_true',
        help='Skip dataset analysis and visualization'
    )
    
    return parser.parse_args()


def main():
    """
    Main function to process data and create LMDB databases.
    """
    args = parse_args()
    
    # Set default output_dir based on target_property if not provided
    if args.output_dir is None:
        args.output_dir = f"set_{args.target_property}_train"
    
    # Validate split ratios
    if abs(sum(args.split_ratios) - 1.0) > 1e-6:
        print("Error: Split ratios must sum to 1.0")
        return
    
    # Target properties to extract
    properties = {
        "material_id": "material_id",
        args.target_property: args.target_property
    }
    
    print("Loading dataset...")
    try:
        compounds_df = pd.read_csv(args.csv_file)
        print(f"Loaded {len(compounds_df)} compounds from {args.csv_file}")
    except FileNotFoundError:
        print(f"Error: CSV file not found: {args.csv_file}")
        return
    
    # Limit dataset size if specified
    if args.max_samples is not None and args.max_samples > 0:
        if args.max_samples < len(compounds_df):
            print(f"Limiting dataset to {args.max_samples} samples "
                  f"(from {len(compounds_df)})")
            
            # Use random sampling for better representation
            if args.seed is not None:
                compounds_df = compounds_df.sample(
                    n=args.max_samples, random_state=args.seed)
            else:
                compounds_df = compounds_df.sample(n=args.max_samples)
            
            # Reset index to maintain consistency
            compounds_df = compounds_df.reset_index(drop=True)
            print(f"Dataset reduced to {len(compounds_df)} samples")
        else:
            print(f"Requested max_samples ({args.max_samples}) >= "
                  f"dataset size ({len(compounds_df)}), using all samples")
    
    # Display basic statistics about the target property
    if args.target_property in compounds_df.columns:
        prop_values = compounds_df[args.target_property].dropna()
        if len(prop_values) > 0:
            print(f"\n{args.target_property} statistics:")
            print(f"  Count: {len(prop_values)}")
            print(f"  Mean: {prop_values.mean():.4f}")
            print(f"  Std: {prop_values.std():.4f}")
            print(f"  Min: {prop_values.min():.4f}")
            print(f"  Max: {prop_values.max():.4f}")
        else:
            print(f"Warning: No valid values found for {args.target_property}")
    else:
        print(f"Warning: Target property '{args.target_property}' "
              f"not found in CSV")
    
    print("\nConverting to atoms list...")
    atoms_list = db_to_atomslist(compounds_df, dir_poscar=args.poscar_dir, 
                                 properties=properties)
    print(f"Successfully converted {len(atoms_list)} structures")
    
    if len(atoms_list) == 0:
        print("No structures were loaded. Check your data paths.")
        return
    
    print("Splitting dataset...")
    split_set = split_dataset(atoms_list, ratio=args.split_ratios,
                              seed=args.seed)
    
    # Print split information
    print(f"{'Set name':<15}{'Set size':>10}")
    print("-" * 25)
    for key, item in split_set.items():
        print(f"{key:<15}{len(item):>10}")
    
    print(f"Creating LMDB databases in {args.output_dir}...")
    split_set_to_lmdb(split_set, properties, dir_name=args.output_dir)
    
    # Validate created LMDB databases
    print("\nValidating created LMDB databases...")
    validation_passed = True
    for split_name in ['train', 'val', 'test']:
        lmdb_path = f"{args.output_dir}/{split_name}.lmdb"
        if not validate_lmdb_database(lmdb_path):
            validation_passed = False
    
    if not validation_passed:
        print("⚠️  Some databases failed validation. Check warnings above.")
    else:
        print("✓ All databases passed validation")
    
    if not args.no_analysis:
        print("Analyzing dataset statistics...")
        
        # Analyze each split
        for split_name in ['train', 'val', 'test']:
            lmdb_path = f"{args.output_dir}/{split_name}.lmdb"
            print(f"\n{split_name.upper()} set statistics:")
            analyze_dataset_statistics(lmdb_path, args.target_property)
    
    print(f"\nDataset preparation complete! "
          f"Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
