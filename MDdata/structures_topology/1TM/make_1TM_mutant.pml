# Build 1TM triple-mutant (S26D + S41D + T63D) from WT 1Nhp6A binary complex.
# Run: pymol -c make_1TM_mutant.pml

load AF_Nhp6A/1Nhp6A_DNA_AF_fix.pdb, wt

wizard mutagenesis

# S26D
python
cmd.get_wizard().set_mode('NO_BUMP')
cmd.get_wizard().do_select('resi 26 and chain A')
cmd.get_wizard().set_residue('ASP')
cmd.get_wizard().apply()
python end

# S41D
python
cmd.get_wizard().set_mode('NO_BUMP')
cmd.get_wizard().do_select('resi 41 and chain A')
cmd.get_wizard().set_residue('ASP')
cmd.get_wizard().apply()
python end

# T63D
python
cmd.get_wizard().set_mode('NO_BUMP')
cmd.get_wizard().do_select('resi 63 and chain A')
cmd.get_wizard().set_residue('ASP')
cmd.get_wizard().apply()
python end

set_wizard

save AF_Nhp6A/1TM_DNA_AF_fix_mutant.pdb, wt

quit
