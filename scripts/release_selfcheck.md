# 发版前自检清单（release-selfcheck）

来源:批 70~76 全部发版验收报告的发现蒸馏(2026-10-26 用户定规矩:
**workflow 验收每版最多跑一次**;跑前先过本清单;发现 high/medium 修一轮
立即发,余账(low/声明类)记入下批候选——多轮验收耗时费 token,不再回头)。

用法:每次触发 release-acceptance workflow 之前,对着本次 diff 逐项过
适用条目;全部通过才准跑 workflow。

## A. 声明与 diff 相符(最高频,占验收发现近半)

1. **计数核对**:提交说明里的一切数字(几处调用点/几条注释/多少项)
   必须与 diff 实际数目一致——`git show --stat` + 目标 grep 复数一遍。
   先例:批71「三处」实为 4 处;批75「两处」漏第三处;批73「~180 条」实 276。
2. **声明的改动必须在 diff 里**:python/批量脚本中途断言失败会把后续
   文件的改动静默吞掉——提交后 `git show --stat` 对着说明逐文件点名。
   先例:批76二修声明改 PeopleView,diff 里没有(断言中断漏改)。
3. **注释的批次号/史实与 git 一致**:写「第 X 批某修」前 grep 确认该
   改动真出自那个提交;拿不准就写提交 sha。
4. **顺序描述与代码顺序一致**:「先取消再收尾」类叙述,读一遍实际行序。
   先例:批76 catch 里 cancel 排在 finishTask 之后=对 nil 无操作。

## B. 类型与编译(macOS CI 是唯一拦截层)

1. **三目/链式拼接拆步**:三目嵌进 `+` 链会让编译器类型推断超时
   (unable to type-check in reasonable time)——一律拆成 if+append,
   数组显式 `[String]` 注解。先例:批72(688 行)、批74 同型复发。
2. **API 归属三查**:属性属于谁(URLRequest vs URLSessionConfiguration)、
   是否 throwing(FileHandle.write(_:) 非 throwing,磁盘满抛 ObjC 异常
   **崩进程**——用 write(contentsOf:))、非标签元组用 `.1` 不用成员名。
3. **本地 -parse 过 ≠ 编译过**:不在本地 typecheck 集(仅
   QueryUnderstanding/LabelZh 系)的文件——Views、IndexingCoordinator、
   VectorStore、QueryTimePlace——每次改动按本节 1/2 人工过一遍。
   先例:QueryTimePlace .remaining、ModelUpdateView timeoutIntervalForResource,
   都是 CI 拦的,parse 全绿。

## C. 数据库与边界态(改 SQL/标记前)

1. **先查约束再改**:动任何列的语义前 grep 建表 UNIQUE 约束与该列的
   全部消费方(含 Views 层查询)。先例:frame_index 被当标记位,但它在
   UNIQUE(kind,ref_key,frame_index) 键里——upsert 0 行不会覆盖 -1 行,
   双行并排=标记永不清除(批74 high)。
2. **UPDATE 撞约束态穷举**:同 refKey 多行并存(部分提交/半途失败产生)
   时 UPDATE 打成同值会整条原子失败——先 DELETE 冲突行再 UPDATE,
   python sqlite3 同构复现含双行态。先例:批75 high。
3. **恰好相等边界**:丢弃条件是 `>` 时,`==` 态走哪条路?必须可达且
   有消费方。先例:.part==size 发 Range: bytes=size- → 416 永久卡死
   (批76 medium)。
4. **非事务序列化**:serialized 是队列锁不是事务,块内多语句可半途
   残留——每条语句独立成立才有无害性论证。

## D. 并发与回调

1. **完成校验用磁盘真相**:Task{@MainActor} 跳转无顺序保证,内存计数
   可能滞后——最终一致性判断读文件实际大小,不读计数(批76 low)。
2. **声明的取消要真生效**:cancel 排在状态清理之后=对 nil 无操作;
   且主动 cancel 的回声回调要吞掉,不吞会误伤下一轮 continuation
   (批76三修 medium)。

## E. 对称面(用户铁律的执行检查)

1. **改绑定/查询要 grep 旧名全库**:修一处 UI 绑定,同名绑定在别的
   View 还有几处?先例:批75 修了 LibraryView 漏 SearchView 横幅。
2. **改裁决语义必须升盐**:归组/识别逻辑的输入集变了(查询放宽/阈值
   改),算法盐同步 +N,否则旧快照复活旧判决;注释同步新盐号
   (批76 a21→a22)。
3. **删守卫后清悬空引用**:grep 被删符号名,注释里的历史引用补上下文
   (批71「上方 guard」悬空)。

## F. 流程纪律

- workflow 验收每版**一次**;跑前自检留痕(本清单哪几条适用、结论);
- 验收后:high/medium → 修一轮(parse+CI 绿)即发;low → 记入
  `批NN 候选`(memory 的批记忆),下批带走;
- 提交说明写「修了什么+先例引用」,不写宣传语——验收员逐句对 diff。
