import os
import shutil

def copy_directory(code, dest):
  try:
    shutil.copytree(code, dest)
  except shutil.Error as e:
    print('Directory not copied. Error: %s' % e)
  except OSError as e:
    print('Directory not copied. Error: %s' % e)

shutil.rmtree('temp_submission', ignore_errors=True)
os.mkdir('temp_submission')
for dir_name in ['code', 'html', 'results']:
  copy_directory(dir_name, '/'.join(['temp_submission', dir_name]))
shutil.make_archive('submission', 'zip', 'temp_submission')
shutil.rmtree('temp_submission', ignore_errors=True)