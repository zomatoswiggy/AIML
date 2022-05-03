
graph={
    'A':['B','C'],
    'B' : ['D', 'E'],
    'C' : ['F'],
    'D' : [],
    'E' : ['F'],
    'F' : []
}

visited=[]
queue=[]

def bfs(visited,graph,node):
    visited.append(node)
    queue.append(node)

    while queue:
        s=queue.pop(0)
        print(s,end=" ")

        for i in graph[s]:
            if i not in visited:
                visited.append(i)
                queue.append(i)






visited_dfs=set()

def dfs(visited_dfs,graph,node):
    if node not in visited:
        print(node,end=" ")
        visited_dfs.append(node)
        for  i in graph[node]:
            dfs(visited_dfs,graph,i)


print(bfs(visited,graph,'A'))
print(dfs(visited,graph,'A'))
# bfs(visited,graph,'A')
# dfs(visited,graph,'A')