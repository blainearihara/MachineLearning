# Python Notes

### \_\_main__  \_\_name__
```python
if __name__ == "__main__":
```
is a Python convention that controls whether a file is being run directly or being imported as a module
- Every Python file has a build-in variable called \_\_name__
- If you run the file directly then Python sets
```Python
__name__ == "__main__"
```
- If you import that file as a module in another script, then \_\_name__ will instead be the module's name

